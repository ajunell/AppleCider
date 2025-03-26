import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import optuna
from scipy import stats

from tqdm.auto import tqdm
from datetime import datetime
import wandb
import numpy as np
import pickle
import joblib

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, LinearLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F

from AppleCider.core.dataset import DataGenerator
from AppleCider.core.model import Informer, GalSpecNet, MetaModel, BTSModel, AppleCider, ZwickyCider
from AppleCider.models.Informer import DataEmbedding, EncoderLayer, AttentionLayer, ProbAttention, Encoder
from AppleCider.util.early_stopping import EarlyStopping


def get_model(config):

    if config['mode'] == 'photo':
        model = Informer(config)
    elif config['mode'] == 'spectra':
        model = GalSpecNet(config)
    elif config['mode'] == 'meta':
        model = MetaModel(config)
    elif config['mode'] == 'image':
        model = BTSModel(config)
    # ztf mode has photometry, images, metadata only
    elif config['mode'] == 'ztf':
        model = ZwickyCider(config)
    
    elif config['mode'] == 'all':
        model = AppleCider(config)
    
    else:
        raise ValueError("no model mode selected!")

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


class Trainer:
    def __init__(self, model, optimizer, scheduler, warmup_scheduler, criterion, criterion_val, device, config, trial=None):
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_scheduler = warmup_scheduler
        self.criterion = criterion
        self.criterion_val = criterion_val
        self.device = device
        self.trial = trial

        self.mode = config['mode']
        self.save_weights = config['save_weights']
        self.weights_path = config['weights_path']
        self.use_wandb = config['use_wandb']
        self.early_stopping = EarlyStopping(patience=config['early_stopping_patience'])
        self.warmup_epochs = config['warmup_epochs']
        self.clip_grad = config['clip_grad']
        self.clip_value = config['clip_value']

        self.total_loss = []
        self.total_correct_predictions = 0
        self.total_predictions = 0
        
        self.custom_weight_path = config['custom_weight_path']
        self.custom_weight_name = config['custom_weight_name']
        
        if self.use_wandb:
            self.run_id = config['run_id']
  
    def store_weights(self, epoch):
        
        if self.use_wandb:
            torch.save(self.model.state_dict(), os.path.join(self.weights_path, f'weights-{datetime.now().strftime("%Y-%m-%d-%H-%M")}-{epoch}-{self.run_id}.pth'))
            torch.save(self.model.state_dict(), os.path.join(self.weights_path, f'weights-{datetime.now().strftime("%Y-%m-%d-%H-%M")}-best-{self.run_id}.pth'))
            
        else:
            if self.custom_weight_path:
                
                torch.save(self.model.state_dict(), os.path.join(self.weights_path, f'weights-{datetime.now().strftime("%Y-%m-%d-%H-%M")}-{epoch}-{self.custom_weight_name}.pth'))
                torch.save(self.model.state_dict(), os.path.join(self.weights_path, f'weights-{datetime.now().strftime("%Y-%m-%d-%H-%M")}-best-{self.custom_weight_name}.pth'))
            
            else:
            
                torch.save(self.model.state_dict(), os.path.join(self.weights_path, f'weights-{datetime.now().strftime("%Y-%m-%d-%H-%M")}-{epoch}-.pth'))
                torch.save(self.model.state_dict(), os.path.join(self.weights_path, f'weights-{datetime.now().strftime("%Y-%m-%d-%H-%M")}-best.pth'))

    def zero_stats(self):
        self.total_loss = []
        self.total_correct_predictions = 0
        self.total_predictions = 0

    # TODO update to 4 elements
    def update_stats_clip(self, loss, logits_ps, logits_sm, logits_mp):
        labels = torch.arange(logits_ps.shape[0], dtype=torch.int64, device=self.device)

        prob_ps = (F.softmax(logits_ps, dim=1) + F.softmax(logits_ps.transpose(-1, -2), dim=1)) / 2
        prob_sm = (F.softmax(logits_sm, dim=1) + F.softmax(logits_sm.transpose(-1, -2), dim=1)) / 2
        prob_mp = (F.softmax(logits_mp, dim=1) + F.softmax(logits_mp.transpose(-1, -2), dim=1)) / 2
        prob = (prob_ps + prob_sm + prob_mp) / 3

        _, pred_labels = torch.max(prob, dim=1)
        correct_predictions = (pred_labels == labels).sum().item()

        self.total_correct_predictions += correct_predictions
        self.total_predictions += labels.size(0)
        self.total_loss.append(loss.item())

    def update_stats(self, loss, logits, labels):
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        _, predicted_labels = torch.max(probabilities, dim=1)
        correct_predictions = (predicted_labels == labels).sum().item()

        self.total_correct_predictions += correct_predictions
        self.total_predictions += labels.size(0)
        self.total_loss.append(loss.item())

    def calculate_stats(self):
        return sum(self.total_loss) / len(self.total_loss), self.total_correct_predictions / self.total_predictions

    def get_logits(self, photometry, photometry_mask, metadata, images, spectra):
        
        if self.mode == 'photo':
            logits = self.model(photometry, photometry_mask)
        elif self.mode == 'spectra':
            logits = self.model(spectra)
        elif self.mode == 'meta':
            logits = self.model(metadata)
        elif self.mode == 'image':
            logits = self.model(images)
        elif self.mode == 'ztf':
            logits = self.model(photometry, photometry_mask, metadata, images)
        else:  # all 4 modalities
            logits = self.model(photometry, photometry_mask, metadata, images, spectra)

        return logits

    # TODO Update to 4 elements
    def step_clip(self, photometry, photometry_mask, spectra, metadata):
        """Perform a training step for the CLIP pretraining model"""
        logits_ps, logits_sm, logits_mp = self.model(photometry, photometry_mask, spectra, metadata)
        
        loss_ps, loss_sm, loss_mp = self.criterion(logits_ps, logits_sm, logits_mp)
        loss = loss_ps + loss_sm + loss_mp

        self.update_stats_clip(loss, logits_ps, logits_sm, logits_mp)

        return loss, loss_ps, loss_sm, loss_mp
    
    def step_val_clip(self, photometry, photometry_mask, spectra, metadata):
        """Perform a training step for the CLIP pretraining model"""
        logits_ps, logits_sm, logits_mp = self.model(photometry, photometry_mask, spectra, metadata)
        
        loss_ps, loss_sm, loss_mp = self.criterion_val(logits_ps, logits_sm, logits_mp)
        loss = loss_ps + loss_sm + loss_mp

        self.update_stats_clip(loss, logits_ps, logits_sm, logits_mp)

        return loss, loss_ps, loss_sm, loss_mp

    def step(self, photometry, photometry_mask, metadata, images, spectra, labels):
        """Perform a training step for the classification model"""
        logits = self.get_logits(photometry, photometry_mask, metadata, images, spectra)    
        
        loss = self.criterion(logits, labels)

        self.update_stats(loss, logits, labels)

        return loss
    
    def step_val(self, photometry, photometry_mask, metadata, images, spectra, labels):
        """Perform a training step for the classification model"""
        logits = self.get_logits(photometry, photometry_mask, metadata, images, spectra)    
        
        loss = self.criterion_val(logits, labels)
        
        self.update_stats(loss, logits, labels)

        return loss

    def get_gradient_norm(self):
        total_norm = 0.0

        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        return total_norm ** 0.5

    def train_epoch(self, train_dataloader):
        self.model.train()
        self.zero_stats()
        
        for photometry, photometry_mask, metadata, images, spectra, labels in tqdm(train_dataloader, total=len(train_dataloader), desc='Train', colour='#9ACD32',leave=True):
            photometry, photometry_mask = photometry.to(self.device), photometry_mask.to(self.device)
            metadata, images, spectra = metadata.to(self.device), images.to(self.device), spectra.to(self.device)
            labels = labels.to(self.device)    
            
            self.optimizer.zero_grad()

            if self.mode == 'clip':
                # TODO add images
                loss, loss_ps, loss_sm, loss_mp = self.step_clip(photometry, photometry_mask, spectra, metadata)
                
                if self.use_wandb:
                    wandb.log({'step_loss': loss.item(), 'loss_ps': loss_ps.item(), 'loss_sm': loss_sm.item(),
                               'loss_mp': loss_mp.item()})
            else:
                loss = self.step(photometry, photometry_mask, metadata, images, spectra, labels)

                if self.use_wandb:
                    wandb.log({'step_loss': loss.item()})

            loss.backward()

            if self.use_wandb:
                grad_norm = self.get_gradient_norm()
                wandb.log({'grad_norm': grad_norm})

            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)

                if self.use_wandb:
                    clip_grad_norm = self.get_gradient_norm()
                    wandb.log({'clip_grad_norm': clip_grad_norm})

            self.optimizer.step()

        loss, acc = self.calculate_stats()

        return loss, acc
   

    def val_epoch(self, val_dataloader):
        self.model.eval()
        self.zero_stats()

        with torch.no_grad():
            for photometry, photometry_mask, metadata, images, spectra, labels in tqdm(val_dataloader, total=len(val_dataloader), desc='Validation', colour='#9ACD32', leave=True):
                photometry, photometry_mask = photometry.to(self.device), photometry_mask.to(self.device)
                metadata, images, spectra = metadata.to(self.device), images.to(self.device), spectra.to(self.device)
                labels = labels.to(self.device)

                if self.mode == 'clip':
                    self.step_val_clip(photometry, photometry_mask, spectra, metadata)
                else:
                    self.step_val(photometry, photometry_mask, metadata, images, spectra, labels)

        loss, acc = self.calculate_stats()

        return loss, acc
    

    def train(self, train_dataloader, val_dataloader, epochs):
        best_val_loss = np.inf
        best_val_acc = 0

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_dataloader)
            val_loss, val_acc = self.val_epoch(val_dataloader)

            best_val_loss = min(val_loss, best_val_loss)

            if self.trial:
                self.trial.report(val_loss, epoch)

                if self.trial.should_prune():
                    print('Prune')
                    wandb.finish()
                    raise optuna.exceptions.TrialPruned()

            if self.warmup_scheduler and epoch < self.warmup_epochs:
                self.warmup_scheduler.step()
                current_lr = self.warmup_scheduler.get_last_lr()[0]
            else:
                self.scheduler.step(val_loss)
                current_lr = self.scheduler.get_last_lr()[0]

            if self.use_wandb:
                wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'train_acc': train_acc,
                           'val_acc': val_acc,'learning_rate': current_lr, 'epoch': epoch})

            if best_val_acc < val_acc:
                best_val_acc = val_acc

                if self.use_wandb:
                    wandb.log({'best_val_acc': best_val_acc})

                if self.save_weights:
                    self.store_weights(epoch)

            print(f'Epoch {epoch}: Train Loss {round(train_loss, 4)} \t Val Loss {round(val_loss, 4)} \t \
                    Train Acc {round(train_acc, 4)} \t Val Acc {round(val_acc, 4)}')

            if self.early_stopping.step(val_loss):
                print(f'Early stopping at epoch {epoch}')
                break

        return best_val_loss

    def evaluate(self, val_dataloader, id2target):
        self.model.eval()

        all_true_labels = []
        all_predicted_labels = []
        
        
        for photometry, photometry_mask, metadata, images, spectra, labels in tqdm(val_dataloader, total=len(val_dataloader), desc='validation', colour='#9ACD32',leave=True):
            with torch.no_grad():
                photometry, photometry_mask = photometry.to(self.device), photometry_mask.to(self.device)
                metadata, images, spectra = metadata.to(self.device), images.to(self.device), spectra.to(self.device)

                logits = self.get_logits(photometry, photometry_mask, metadata, images, spectra)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                _, predicted_labels = torch.max(probabilities, dim=1)

                all_true_labels.extend(labels.numpy())
                all_predicted_labels.extend(predicted_labels.cpu().numpy())

        conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)
        conf_matrix_percent = 100 * conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]

        labels = [id2target[i] for i in range(len(conf_matrix))]
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(40, 15))

        sns.set(font_scale=2.7) 
        label_font = {'size':'35'}
        
        axes[0].tick_params(labelsize=25)
        # Plot absolute values confusion matrix
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='BuPu', xticklabels=labels, yticklabels=labels, ax=axes[0])
        axes[0].set_xlabel('Predicted', fontdict=label_font)
        axes[0].set_ylabel('True',  fontdict=label_font)
        axes[0].set_title('Confusion Matrix - Absolute Values', fontsize=35)

        # Plot percentage values confusion matrix
        sns.heatmap(conf_matrix_percent, annot=True, fmt='.0f', cmap='BuPu', xticklabels=labels, yticklabels=labels,
            ax=axes[1])
        axes[1].tick_params(labelsize=25)
        axes[1].set_xlabel('Predicted', fontdict=label_font)
        axes[1].set_title('Confusion Matrix - %', fontsize=35)

        if self.use_wandb:
            wandb.log({'conf_matrix': wandb.Image(fig)})
            
            artifact_m = wandb.Artifact("matrix", type="dataset")
            matrix_table = wandb.Table(columns=["SN Ia","SN Ic","SN Ib", "SN II", "SN IIP", "SN IIn", "SN IIb", "CV", "AGN", "TDE"], data=conf_matrix)
            artifact_m.add(matrix_table, "(matrix) confusion matrix")
            wandb.log_artifact(artifact_m)

        return conf_matrix
    
    
    
def collate_func(data):
    photometry, metadata, images, spectra, labels = zip(*data)
    
    labels = torch.tensor(labels, dtype=torch.int64)
    
    photometry = torch.stack(photometry)
    photometry_mask = torch.ones((photometry.size(0), photometry.size(1)))
    metadata = torch.stack(metadata)
    images = torch.stack(images)
    spectra = torch.stack(spectra)
    
    return photometry, photometry_mask, metadata, images, spectra, labels
    
    
def run(config, add_notes=None, tags_list=None):
    
    if config['use_wandb']:
        
        if config['use_notes_tags']:
        
            wandb_run = wandb.init(project=config['project'], config=config, notes= config['wandb_notes'], tags=config['wandb_tags'])
            config['run_id'] = wandb_run.id
            config.update(wandb.config)
            print('run name',wandb_run.name)
            print('run id', wandb_run.id)

        else:
            wandb_run = wandb.init(project=config['project'], config=config)
            config['run_id'] = wandb_run.id
            config.update(wandb.config)
            print('run name',wandb_run.name)
            print('run id', wandb_run.id)
    
    train_dataset = DataGenerator(config, split='train')
    val_dataset = DataGenerator(config, split='val')
    
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, collate_fn=collate_func)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], collate_fn=collate_func, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', device)
    
    model = get_model(config)
    model = model.to(device)
    
    if config['use_wandb']:
        wandb.log({'model': model})
        
    optimizer = Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2'] ))
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-5, end_factor=1, total_iters=config['warmup_epochs'])
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['factor'], patience=config['patience'])
    
    if config['class_weights']:
        if os.path.isfile(config['class_weights_path']):
            ## weights
            with open(config['class_weights_path'], 'rb') as file:
                weights = pickle.load(file)
            weight_sorted = dict(sorted(weights.items(), key=lambda item: item))
            weight_sorted_list = list(weight_sorted.values())
            weight_tensor = torch.tensor(weight_sorted_list, dtype=torch.float32)
            criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
            criterion_val = torch.nn.CrossEntropyLoss()
            
        else:
            raise ValueError(f'class weights=True, but no class weight file at config[class_weights_path] exists')
        
    else:
        criterion = torch.nn.CrossEntropyLoss()
        criterion_val = torch.nn.CrossEntropyLoss()
    
    trainer = Trainer(model=model, optimizer=optimizer, scheduler=scheduler, warmup_scheduler=warmup_scheduler, criterion=criterion, criterion_val = criterion_val, device=device, config=config)
    trainer.train(train_dataloader, val_dataloader, epochs=config['epochs'])
    
    if config['mode'] != 'clip':
        trainer.evaluate(val_dataloader, id2target=train_dataset.target2id) 

