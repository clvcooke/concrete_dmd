import wandb
import dataloader
from network import DigitNet
from trainer import FixedTrainer

wandb.init("cdmd")

if __name__ == "__main__":
    # TODO: get task, read args, add adaptive
    train_loader, val_loader = dataloader.load_train_data('mnist', batch_size=256)
    network = DigitNet(dmd_count=2)
    trainer = FixedTrainer(network, train_loader, val_loader)
    trainer.train()
