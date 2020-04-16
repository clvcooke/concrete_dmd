import wandb
import dataloader
from network import FixedDigitNet, AdaptiveDigitNet
from trainer import FixedTrainer, AdaptiveTrainer

wandb.init("cdmd")

if __name__ == "__main__":
    # TODO: get task, read args, add adaptive
    train_loader, val_loader = dataloader.load_train_data('mnist', batch_size=256)
    adaptive = True
    if not adaptive:
        network = FixedDigitNet(dmd_count=2, temperature=1.5)
        trainer = FixedTrainer(network, train_loader, val_loader, init_lr=1e-3)
    else:
        network = AdaptiveDigitNet(dmd_count=2)
        trainer = AdaptiveTrainer(network, train_loader, val_loader, init_lr=1e-3)
    trainer.train()

