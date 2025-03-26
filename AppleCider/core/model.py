import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, LinearLR

from datetime import datetime
import optuna
from optuna.exceptions import DuplicatedStudyError
from tqdm.auto import tqdm


from AppleCider.core.dataset import DataGenerator
from AppleCider.core.trainer import Trainer
from AppleCider.models.Informer import DataEmbedding, EncoderLayer, AttentionLayer, ProbAttention, Encoder


class Informer(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132
    """
    def __init__(self, config):
        super(Informer, self).__init__()

        self.classification = True if config['mode'] == 'photo' else False
        self.enc_in = config['p_enc_in']
        self.d_model = config['p_d_model']
        self.dropout = config['p_dropout']
        self.factor = config['p_factor']
        self.output_attention = config['p_output_attention']
        self.n_heads = config['p_n_heads']
        self.d_ff = config['p_d_ff']
        self.activation = config['p_activation']
        self.e_layers = config['p_e_layers']

        self.enc_embedding = DataEmbedding(self.enc_in, self.d_model)

        attn_layers = [
            EncoderLayer(
                AttentionLayer(
                    ProbAttention(False, self.factor, attention_dropout=self.dropout,
                                  output_attention=self.output_attention),
                    self.d_model,
                    self.n_heads
                ),
                self.d_model,
                self.d_ff,
                dropout=self.dropout,
                activation=self.activation
            ) for _ in range(self.e_layers)
        ]
        self.encoder = Encoder(attn_layers, norm_layer=torch.nn.LayerNorm(self.d_model))

        self.dropout = nn.Dropout(self.dropout)

        if self.classification:
            self.fc = nn.Linear(config['seq_len'] * config['p_d_model'], config['num_classes'])

    def forward(self, x_enc, x_mark_enc):
       
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
    
        enc_out = self.dropout(enc_out)
        
        output = enc_out * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)

        if self.classification:
            output = self.fc(output)

        return output


class GalSpecNet(nn.Module):
    
    """
    GalSpecNet
    Paper link: https://doi.org/10.1093/mnras/stad2913
    """
    
    def __init__(self, config):
        super(GalSpecNet, self).__init__()

        self.classification = True if config['mode'] == 'spectra' else False
        self.dropout_rate = config['s_dropout']

        self.conv_channels = config['s_conv_channels']
        self.kernel_size = config['s_kernel_size']
        self.mp_kernel_size = config['s_mp_kernel_size']

        self.layers = nn.ModuleList([])

        for i in range(len(self.conv_channels) - 1):
            self.layers.append(
                nn.Conv1d(self.conv_channels[i], self.conv_channels[i + 1], kernel_size=self.kernel_size)
            )
            self.layers.append(nn.ReLU())

            if i < len(self.conv_channels) - 2:  # Add MaxPool after each Conv-ReLU pair except the last
                self.layers.append(nn.MaxPool1d(kernel_size=self.mp_kernel_size))

        self.dropout = nn.Dropout(self.dropout_rate)

        if self.classification:
            self.fc = nn.Linear(1632, config['num_classes'])

    def forward(self, x):
        
        for layer in self.layers:
            x = layer(x)

        x = x.view(x.shape[0], -1)
        x = self.dropout(x)

        if self.classification:
            x = self.fc(x)

        return x


class MetaModel(nn.Module):
    
    """
    Metadata model from AstroM3
    Paper link: https://arxiv.org/abs/2411.08842
    """""
    
    def __init__(self, config):
        super(MetaModel, self).__init__()

        self.classification = True if config['mode'] == 'meta' else False
        self.input_dim = len(config['meta_cols'])
        self.hidden_dim = config['m_hidden_dim']
        self.dropout = config['m_dropout']

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        if self.classification:
            self.fc = nn.Linear(self.hidden_dim, config['num_classes'])

    def forward(self, x):
        
        x = self.model(x)

        if self.classification:
            x = self.fc(x)

        return x


class BTSModel(nn.Module):
    
    """
    Image model from BTSbot for science, reference, difference images 
    Paper link: https://arxiv.org/abs/2401.15167
    """""
    
    def __init__(self, config):
        super(BTSModel, self).__init__()

        self.classification = True if config['mode'] == 'image' else False

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=config['input_channels'],
                out_channels=config['conv1_channels'],
                kernel_size=config['conv_kernel'],
                padding='same'
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config['conv1_channels'],
                out_channels=config['conv1_channels'],
                kernel_size=config['conv_kernel'],
                padding='same'
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(config['conv_dropout1'])
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=config['conv1_channels'],
                out_channels=config['conv2_channels'],
                kernel_size=config['conv_kernel'],
                padding='same'
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config['conv2_channels'],
                out_channels=config['conv2_channels'],
                kernel_size=config['conv_kernel'],
                padding='same'
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.Dropout(config['conv_dropout2'])
        )

        if self.classification:
            self.fc = nn.Linear(784, config['num_classes'])

    def forward(self, x):
        
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.shape[0], -1)

        if self.classification:
            x = self.fc(x)

        return x


class AppleCider(nn.Module):
    
    """
    AppleCider = AstroM3 (photometry, metadata, spectra) + BTSbot (images)
    """
    
    def __init__(self, config):
        super(AppleCider self).__init__()

        self.classification = True if config['mode'] == 'all' else False

        self.photometry_encoder = Informer(config)
        self.spectra_encoder = GalSpecNet(config)
        self.metadata_encoder = MetaModel(config)
        self.image_encoder = BTSModel(config)

        self.photometry_proj = nn.Linear(config['seq_len'] * config['p_d_model'], config['hidden_dim'])
        
        self.spectra_proj = nn.Linear(1632, config['hidden_dim']) 
        
        self.metadata_proj = nn.Linear(config['m_hidden_dim'], config['hidden_dim'])

        self.image_proj = nn.Linear(784, config['hidden_dim']) 

        # TODO add other logits
        self.logit_scale_ps = nn.Parameter(torch.log(torch.ones([]) * 100))
        self.logit_scale_sm = nn.Parameter(torch.log(torch.ones([]) * 100))
        self.logit_scale_mp = nn.Parameter(torch.log(torch.ones([]) * 100))

        if self.classification:
            self.fusion = config['fusion']
            in_features = config['hidden_dim'] * 4 if self.fusion == 'concat' else config['hidden_dim']
            self.fc = nn.Linear(in_features, config['num_classes'])

    def get_embeddings(self, photometry, photometry_mask, metadata, images, spectra):
    
        p_emb = self.photometry_proj(self.photometry_encoder(photometry, photometry_mask))
        s_emb = self.spectra_proj(self.spectra_encoder(spectra))
        m_emb = self.metadata_proj(self.metadata_encoder(metadata))
        i_emb = self.image_proj(self.image_encoder(images))

        # normalize features
        p_emb = p_emb / p_emb.norm(dim=-1, keepdim=True)
        s_emb = s_emb / s_emb.norm(dim=-1, keepdim=True)
        m_emb = m_emb / m_emb.norm(dim=-1, keepdim=True)
        i_emb = i_emb / i_emb.norm(dim=-1, keepdim=True)
        
        return p_emb, m_emb, i_emb, s_emb

    def forward(self, photometry, photometry_mask, metadata, images, spectra):
        
        p_emb, m_emb, i_emb, s_emb = self.get_embeddings(photometry, photometry_mask, metadata, images, spectra)

        if self.classification:

            if self.fusion == 'concat':
                emb = torch.cat((p_emb, m_emb, i_emb, s_emb), dim=1)
            elif self.fusion == 'avg':
                emb = (p_emb + m_emb + i_emb + s_emb) / 4
            else:
                raise NotImplementedError

            logits = self.fc(emb)

            return logits
        else:
            # TODO change this logic to work with 4 elements
            raise NotImplementedError
            # logit_scale_ps = torch.clamp(self.logit_scale_ps.exp(), min=1, max=100)
            # logit_scale_sm = torch.clamp(self.logit_scale_sm.exp(), min=1, max=100)
            # logit_scale_mp = torch.clamp(self.logit_scale_mp.exp(), min=1, max=100)
            #
            # logits_ps = logit_scale_ps * p_emb @ s_emb.T
            # logits_sm = logit_scale_sm * s_emb @ m_emb.T
            # logits_mp = logit_scale_mp * m_emb @ p_emb.T
            #
            # return logits_ps, logits_sm, logits_mp
            
            
            
class ZwickyCider(nn.Module):
    
    """
    ZwickyCider = AppleCider - spectra.
    for ZTF classification purposes
    """
    
    def __init__(self, config):
        super(ZwickyCider, self).__init__()

        self.classification = True if config['mode'] == 'ztf' else False

        self.photometry_encoder = Informer(config)
        self.metadata_encoder = MetaModel(config)
        self.image_encoder = BTSModel(config)

        self.photometry_proj = nn.Linear(config['seq_len'] * config['p_d_model'], config['hidden_dim'])
        self.metadata_proj = nn.Linear(config['m_hidden_dim'], config['hidden_dim'])
        self.image_proj = nn.Linear(784, config['hidden_dim']) 

        self.logit_scale_pi = nn.Parameter(torch.log(torch.ones([]) * 100))
        self.logit_scale_im = nn.Parameter(torch.log(torch.ones([]) * 100))
        self.logit_scale_mp = nn.Parameter(torch.log(torch.ones([]) * 100))

        if self.classification:
            self.fusion = config['fusion']
            in_features = config['hidden_dim'] * 3 if self.fusion == 'concat' else config['hidden_dim']
            self.fc = nn.Linear(in_features, config['num_classes'])

    def get_embeddings(self, photometry, photometry_mask, metadata, images):
    
        p_emb = self.photometry_proj(self.photometry_encoder(photometry, photometry_mask))
        m_emb = self.metadata_proj(self.metadata_encoder(metadata))
        i_emb = self.image_proj(self.image_encoder(images))

        # normalize features
        p_emb = p_emb / p_emb.norm(dim=-1, keepdim=True)
        m_emb = m_emb / m_emb.norm(dim=-1, keepdim=True)
        i_emb = i_emb / i_emb.norm(dim=-1, keepdim=True)
        
        return p_emb, m_emb, i_emb

    def forward(self, photometry, photometry_mask, metadata, images):
        
        p_emb, m_emb, i_emb = self.get_embeddings(photometry, photometry_mask, metadata, images)

        if self.classification:
            if self.fusion == 'concat':
                emb = torch.cat((p_emb, m_emb, i_emb), dim=1)
            elif self.fusion == 'avg':
                emb = (p_emb + m_emb + i_emb) / 3
            else:
                raise NotImplementedError

            logits = self.fc(emb)

            return logits
        
        else:
            logit_scale_pi = torch.clamp(self.logit_scale_pi.exp(), min=1, max=100)
            logit_scale_im = torch.clamp(self.logit_scale_im.exp(), min=1, max=100)
            logit_scale_mp = torch.clamp(self.logit_scale_mp.exp(), min=1, max=100)
            
            logits_pi = logit_scale_pi * p_emb @ i_emb.T
            logits_im = logit_scale_im * i_emb @ m_emb.T
            logits_mp = logit_scale_mp * m_emb @ p_emb.T
            
            return logits_pi, logiti_sm, logits_mp
