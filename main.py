import wandb
import dataloader
from network import FixedDigitNet, AdaptiveDigitNet
from trainer import FixedTrainer, AdaptiveTrainer, AnnealingTrainer
from dmd import ContinuousDMD
from config import get_config
import torch

wandb.init("cdmd")

VERSION = 2


def main(config):
    torch.manual_seed(config.random_seed)
    torch.set_num_threads(1)
    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)
    train_loader, val_loader = dataloader.load_train_data(config.task, batch_size=config.batch_size)
    adaptive = config.adaptive
    if adaptive:
        network_cls = AdaptiveDigitNet
        trainer_cls = AdaptiveTrainer
    else:
        network_cls = FixedDigitNet
        trainer_cls = FixedTrainer
    network = network_cls(dmd_count=config.num_patterns, temperature=config.temp, hidden_size=config.hidden_size)
    trainer = trainer_cls(network, train_loader, val_loader, init_lr=config.init_lr, epochs=config.epochs)
    wandb.config.update(config)
    wandb.config.update({
        "version": VERSION
    })
    trainer.train()


# if continuou:
#     network = FixedDigitNet(dmd_count=2, temperature=1.5, dmd_type=ContinuousDMD)
#     trainer = AnnealingTrainer(network, train_loader, val_loader, init_lr=1e-3)
# else:

if __name__ == "__main__":
    conf, unparsed = get_config()
    main(conf)
