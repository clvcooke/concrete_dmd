from torch.optim.adamw import AdamW
from tqdm import tqdm
from utils import AverageMeter
import torch.nn.functional as F
import numpy as np
import torch
import wandb
import os


class Trainer:
    def __init__(self, network, train_loader, val_loader, epochs, use_gpu):
        self.use_gpu = use_gpu
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.network = network
        self.run_name = os.path.basename(wandb.run.path)
        self.optimizer = None

    def run_one_epoch(self, loader, curr_epoch, training=True):
        pbar = tqdm(loader, total=len(loader))
        avg_loss = AverageMeter()
        avg_acc = AverageMeter()
        for data, target in pbar:
            if self.use_gpu:
                data, target = data.cuda(), target.cuda()
            output = self.network(data, cold=not training)
            loss = F.nll_loss(output, target)
            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            acc = torch.sum(output.detach().argmax(dim=1) == target.detach()).float() / len(data)
            avg_acc.update(acc)
            avg_loss.update(loss.item())
            pbar.set_description(f'Epoch {curr_epoch} - acc: {avg_acc.avg:.4f} - loss {avg_loss.avg:.4f}')
        return avg_loss.avg, avg_acc.avg


class AdaptiveTrainer(Trainer):
    def __init__(self, network, train_loader, val_loader, init_lr=3e-4, epochs=10, use_gpu=True):
        super().__init__(network, train_loader, val_loader, epochs, use_gpu)
        if use_gpu:
            network.cuda()
            network.first_dmd.cuda()
        parameters = list(self.network.parameters()) + list(self.network.first_dmd.parameters())
        self.optimizer = AdamW(parameters, lr=init_lr)

    def train(self):
        for epoch in range(self.epochs):
            # train
            loss, acc = self.run_one_epoch(self.train_loader, epoch, True)
            # validate
            val_loss, val_acc = self.run_one_epoch(self.val_loader, epoch, False)
            # log
            wandb.log({
                'train_loss': loss,
                'train_acc': acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }, step=epoch + 1)


class FixedTrainer(Trainer):
    def __init__(self, network, train_loader, val_loader, init_lr=3e-4, epochs=10, use_gpu=True):
        super().__init__(network, train_loader, val_loader, epochs, use_gpu)
        if use_gpu:
            self.network.cuda()
            [dmd.cuda() for dmd in self.network.dmds]
        parameters = list(self.network.parameters())
        for dmd in self.network.dmds:
            parameters += list(dmd.parameters())
        self.optimizer = AdamW(parameters, lr=init_lr)

    def train(self):
        self.record_logits(0)
        for epoch in range(self.epochs):
            # train
            loss, acc = self.run_one_epoch(self.train_loader, epoch, True)
            # validate
            val_loss, val_acc = self.run_one_epoch(self.val_loader, epoch, False)
            # log
            wandb.log({
                'train_loss': loss,
                'train_acc': acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }, step=epoch + 1)
            self.record_logits(epoch + 1)

    def record_logits(self, step, directory='./'):
        """
        This function will log the logits of the current DMDs to wandb and disk
        :return:
        """
        logits = [dmd.logits.cpu().detach() for dmd in self.network.dmds]
        save_dir = os.path.join(directory, 'logits', self.run_name, f'epoch_{step}')
        os.makedirs(save_dir, exist_ok=True)
        for i, logit in enumerate(logits):
            np.save(os.path.join(save_dir, f'pattern_{i}.npy'), logit)
        # save the logits to disk based on the wandb run id
        # TODO: change shape to be a variable
        dmd_probs = [F.softmax(logit, dim=-1).reshape(28, 28) for logit in logits]
        images = [wandb.Image(probs * 255, caption=f"DMD {i + 1}") for i, probs in enumerate(dmd_probs)]
        wandb.log({
            'logits': images
        }, step=step)
