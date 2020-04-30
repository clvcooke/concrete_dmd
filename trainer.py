from torch.optim.adamw import AdamW
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from tqdm import tqdm
from utils import AverageMeter
import torch.nn.functional as F
import numpy as np
import torch
import wandb
import os
import kornia


class Trainer:
    def __init__(self, network, train_loader, val_loader, epochs, use_gpu, criterion, init_lr):
        self.use_gpu = use_gpu
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.network = network
        self.run_name = os.path.basename(wandb.run.path)
        self.criterion = criterion
        self.lr = init_lr
        self.classification = criterion == F.nll_loss
        if use_gpu:
            self.network.cuda()
        self.dmd_optim = Adam(self.network.dmds.parameters(), lr=0.001)
        self.optimizer = Adam(self.network.parameters(), lr=init_lr)
        sqrt = int(np.sqrt(self.network.input_size))
        self.logit_shape = (sqrt, sqrt)

    def train(self):
        self.record_logits(0)
        best_val_loss = None
        time_since_last_improvement = 0
        for epoch in range(self.epochs):
            # train
            metrics = self.run_one_epoch(self.train_loader, epoch, True)
            # validate
            metrics.update(self.run_one_epoch(self.val_loader, epoch, False))
            if best_val_loss is None or best_val_loss > metrics['train_loss']:
                old_loss = best_val_loss
                best_val_loss = metrics['val_loss']
                time_since_last_improvement = 0
                print(f"val loss improved from {old_loss} to {best_val_loss}")
            else:
                time_since_last_improvement += 1
            if time_since_last_improvement > 10:
                time_since_last_improvement = 0
                self.lr = self.lr / np.sqrt(10)
                print(f"Reducing LR to {self.lr}")
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
                for param_group in self.dmd_optim.param_groups:
                    param_group['lr'] = self.lr
            self.record_logits(epoch + 1)
            wandb.log(metrics, step=epoch + 1)

    def run_one_epoch(self, loader, curr_epoch, training=True):
        pbar = tqdm(loader, total=len(loader))
        avg_loss = AverageMeter()
        avg_acc = AverageMeter()
        avg_ssim = AverageMeter()
        for data, target in pbar:
            if self.use_gpu:
                data, target = data.cuda(), target.cuda()
            output = self.network(data, cold=not training)
            # output = self.network(data)
            if self.classification:
                loss = self.criterion(output, target)
            else:
                loss = self.criterion(output, data)
            if training:
                self.optimizer.zero_grad()
                self.dmd_optim.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.dmd_optim.step()
            avg_loss.update(loss.item())
            if self.classification:
                acc = torch.sum(output.detach().argmax(dim=1) == target.detach()).float() / len(data)
                avg_acc.update(acc)
                desc = f'Epoch {curr_epoch} - acc: {avg_acc.avg:.4f} - loss {avg_loss.avg:.4f}'
            else:
                ssim = 1 - kornia.losses.ssim(data, output, 11, reduction='mean').item()
                # ssim = measure.compare_ssim(data.cpu(), output.cpu())
                avg_ssim.update(ssim)
                desc = f'Epoch {curr_epoch}  - ssim: {avg_ssim.avg:.4f} - loss {avg_loss.avg:.4f}'
            pbar.set_description(desc)
        if not training and not self.classification:
            self.record_images(curr_epoch, output, data)
        return self.create_metrics(training, avg_loss.avg, avg_acc.avg, avg_ssim.avg, self.classification)

    @staticmethod
    def create_metrics(training, loss, acc=None, ssim=None, classification=True):
        metric_type = 'train' if training else 'val'
        if classification:
            metrics = {
                f"{metric_type}_loss": loss,
                f"{metric_type}_acc": acc
            }
        else:
            metrics = {
                f"{metric_type}_loss": loss,
                f"{metric_type}_ssim": ssim
            }
        return metrics

    def record_logits(self, step, directory='./'):
        """
        This function will log the logits of the current DMDs to wandb and disk
        :return:
        """
        logits = self.network.dmds.logits.cpu().detach()
        save_dir = os.path.join(directory, 'logits', self.run_name, f'epoch_{step}')
        os.makedirs(save_dir, exist_ok=True)
        # for i in range(logits.shape[0]):
        #     np.save(os.path.join(save_dir, f'pattern_{i}.npy'), logits[i, :])
        # save the logits to disk based on the wandb run id
        # TODO: change shape to be a variable
        odds = torch.exp(logits)
        dmd_probs = odds / (1 + odds)
        images = []
        for i in range(min(4, dmd_probs.shape[0])):
            probs_image = np.int0(dmd_probs[i, :].reshape(self.logit_shape) * 255)
            image = wandb.Image(probs_image, caption=f"DMD {i + 1}")
            images.append(image)
        wandb.log({
            'sampling_probs': images
        }, step=step, commit=False)

    @staticmethod
    def record_images(step, output, target):
        # take the first 4 always
        images = [wandb.Image(output[i, 0, :, :] * 255, caption=f"Val Image {i + 1}") for i in range(4)]
        targets = [wandb.Image(target[i, 0, :, :] * 255, caption=f"Val Image {i + 1}") for i in range(4)]
        wandb.log(
            {
                'val_predictions': images,
                'val_targets': targets
            }, step=step, commit=False
        )


class ReconTrainer(Trainer):
    def __init__(self, network, train_loader, val_loader, epochs, use_gpu, init_lr):
        super().__init__(network, train_loader, val_loader, epochs, use_gpu, criterion=F.l1_loss, init_lr=init_lr)


class AnnealingClassificationTrainer(Trainer):
    def __init__(self, network, train_loader, val_loader, init_lr=3e-4, epochs=10, use_gpu=True, criterion=F.nll_loss):
        super().__init__(network, train_loader, val_loader, epochs, use_gpu, criterion, init_lr=init_lr)

    def run_one_epoch(self, loader, curr_epoch, training=True):
        pbar = tqdm(loader, total=len(loader))
        avg_loss = AverageMeter()
        avg_acc = AverageMeter()
        for data, target in pbar:
            if self.use_gpu:
                data, target = data.cuda(), target.cuda()
            output = self.network(data, cold=not training)
            loss = F.nll_loss(output, target)
            # lets add a penalty for diverging from 1 or -1
            divergence_loss = None
            for dmd in self.network.dmds:
                dmd_params = torch.abs(dmd.mask)
                divergence = F.mse_loss(torch.ones_like(dmd_params), dmd_params)
                if divergence_loss is None:
                    divergence_loss = divergence
                else:
                    divergence_loss += divergence
            # loss += divergence_loss
            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            acc = torch.sum(output.detach().argmax(dim=1) == target.detach()).float() / len(data)
            avg_acc.update(acc)
            avg_loss.update(loss.item())
            pbar.set_description(f'Epoch {curr_epoch} - acc: {avg_acc.avg:.4f} - loss {avg_loss.avg:.4f}')
        return avg_loss.avg, avg_acc.avg


class AdaptiveClassificationTrainer(Trainer):
    def __init__(self, network, train_loader, val_loader, init_lr=3e-4, epochs=10, use_gpu=True, criterion=F.nll_loss):
        super().__init__(network, train_loader, val_loader, epochs, use_gpu, criterion, init_lr=init_lr)


class FixedClassificationTrainer(Trainer):
    def __init__(self, network, train_loader, val_loader, init_lr=3e-4, epochs=10, use_gpu=True, criterion=F.nll_loss):
        super().__init__(network, train_loader, val_loader, epochs, use_gpu, criterion, init_lr=init_lr)
