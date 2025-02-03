import os
import wandb
import random
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, LinearLR
from torch.utils.data import DataLoader
from datetime import datetime

from dataset import DataGenerator
from model import Informer, GalSpecNet, MetaModel, BTSModel, AstroM4
from loss import CLIPLoss
from trainer import Trainer

CLASSES = ['SN Ia', 'SN II', 'SN IIP', 'Cataclysmic', 'AGN', 'SN IIn', 'SN Ic', 'SN Ib', 'SN IIb', 'Tidal Disruption Event']


def get_model(config):
    if config['mode'] == 'photo':
        model = Informer(config)
    elif config['mode'] == 'spectra':
        model = GalSpecNet(config)
    elif config['mode'] == 'meta':
        model = MetaModel(config)
    elif config['mode'] == 'image':
        model = BTSModel(config)
    else:
        model = AstroM4(config)

    # TODO Add image
    if config['use_pretrain'] and config['use_pretrain'].startswith('CLIP'):
        weights = torch.load(config['use_pretrain'][4:], weights_only=True)

        if config['mode'] == 'photo':
            weights_prefix = 'photometry_encoder'
        elif config['mode'] == 'spectra':
            weights_prefix = 'spectra_encoder'
        elif config['mode'] == 'meta':
            weights_prefix = 'metadata_encoder'
        else:
            weights_prefix = None

        if weights_prefix:
            weights = {k[len(weights_prefix) + 1:]: v for k, v in weights.items() if k.startswith(weights_prefix)}

        model.load_state_dict(weights, strict=False)
        print('Loaded weights from {}'.format(config['use_pretrain']))

    return model


def get_schedulers(config, optimizer):
    if config['scheduler'] == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer, gamma=config['gamma'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['factor'], patience=config['patience'])
    else:
        raise NotImplementedError(f"Scheduler {config['scheduler']} not implemented")

    if config['warmup']:
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-5, end_factor=1, total_iters=config['warmup_epochs'])
    else:
        warmup_scheduler = None

    return scheduler, warmup_scheduler


def run(config):
    train_dataset = DataGenerator(config, split='train')
    val_dataset = DataGenerator(config, split='val')

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True,
                                  num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', device)

    model = get_model(config)
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']),
                     weight_decay=config['weight_decay'])
    scheduler, warmup_scheduler = get_schedulers(config, optimizer)
    criterion = CLIPLoss() if config['mode'] == 'clip' else torch.nn.CrossEntropyLoss()

    trainer = Trainer(model=model, optimizer=optimizer, scheduler=scheduler, warmup_scheduler=warmup_scheduler,
                      criterion=criterion, device=device, config=config)
    trainer.train(train_dataloader, val_dataloader, epochs=config['epochs'])

    if config['mode'] != 'clip':
        trainer.evaluate(val_dataloader, id2target=train_dataset.id2target)


def set_random_seeds(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_config():
    config = {
        'project': 'TransientsUMN',
        'mode': 'all',    # 'clip' 'photo' 'spectra' 'meta' 'image' 'all'
        'config_from': None,    # 'meridk/AstroCLIPResults/d2u52yml',
        'random_seed': 42,  # 42, 66, 0, 12, 123
        'use_wandb': True,
        'save_weights': False,
        'weights_path': f'/data/dev/ml_skyportal/AppleCider/AstroM3/weights/{datetime.now().strftime("%Y-%m-%d-%H-%M")}',
        # 'use_pretrain': 'CLIP/home/mariia/AstroML/weights/2024-08-14-14-05-zmjau1cu/weights-51.pth',
        'use_pretrain': None,
        'freeze': False,

        # Data General
        'preprocessed_path': '/data/dev/ml_skyportal/AppleCider/data_train_redux/',
        'spectra_path': '/data/dev/ml_skyportal/AJs_Stuff/(aj)data_all/',
        'df_path': '/data/dev/ml_skyportal/AppleCider/data_train.csv',
        'step': 'type',
        'classes': CLASSES,
        'group_labels': True,
        # TODO if labels are grouped max samples should be on a grouped label?
        'max_samples': 5000,
        'num_classes': len(CLASSES),

        # Photometry Model
        'seq_len': 180,
        'p_enc_in': 2,
        'p_d_model': 128,
        'p_dropout': 0.2,
        'p_factor': 1,
        'p_output_attention': False,
        'p_n_heads': 4,
        'p_d_ff': 512,
        'p_activation': 'gelu',
        'p_e_layers': 8,

        # Spectra Model
        's_dropout': 0.2,
        's_conv_channels': [1, 64, 64, 32, 32],
        's_kernel_size': 3,
        's_mp_kernel_size': 4,

        # Metadata Model
        'm_hidden_dim': 256,
        'm_dropout': 0.2,
        # TODO use actual column names
        'meta_cols': range(10),
        'scaler_path': '/data/dev/ml_skyportal/AppleCider/AstroM3/core/scaler.pkl',

        # Image Model
        'input_channels': 3,
        'conv1_channels': 64,
        'conv2_channels': 16,
        'conv_kernel': 3,
        'conv_dropout1': 0.45,
        'conv_dropout2': 0.65,

        # MultiModal Model
        'hidden_dim': 128,
        'fusion': 'avg',  # 'avg', 'concat'

        # Training
        'batch_size': 512,
        'lr': 0.001,
        'beta1': 0.9,
        'beta2': 0.999,
        'weight_decay': 0.01,
        'epochs': 100,
        'early_stopping_patience': 6,
        'scheduler': 'ReduceLROnPlateau',  # 'ExponentialLR', 'ReduceLROnPlateau'
        'gamma': 0.9,  # for ExponentialLR scheduler
        'factor': 0.3,  # for ReduceLROnPlateau scheduler
        'patience': 3,  # for ReduceLROnPlateau scheduler
        'warmup': False,
        'warmup_epochs': 10,
        'clip_grad': False,
        'clip_value': 5
    }

    if config['config_from']:
        print(f"Copying params from the {config['config_from']} run")
        old_config = wandb.Api().run(config['config_from']).config

        for el in old_config:
            if el in [
                'p_dropout', 's_dropout', 'm_dropout', 'lr', 'beta1', 'weight_decay', 'epochs',
                'early_stopping_patience', 'factor', 'patience', 'warmup', 'warmup_epochs', 'clip_grad', 'clip_value',
                'use_pretrain', 'freeze', 'phased', 'p_aux', 's_aux', 's_err', 'file'
            ]:
                config[el] = old_config[el]

    return config


def main():
    config = get_config()
    set_random_seeds(config['random_seed'])

    if config['use_wandb']:
        wandb_run = wandb.init(project=config['project'], config=config)
        config.update(wandb.config)

        config['run_id'] = wandb_run.id
        config['weights_path'] += f'-{wandb_run.id}'
        print(wandb_run.name, config)

    if config['save_weights']:
        os.makedirs(config['weights_path'], exist_ok=True)

    run(config)
    wandb.finish()


if __name__ == '__main__':
    main()
