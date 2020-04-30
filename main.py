import wandb
import dataloader
from network import FixedDigitNet, AdaptiveDigitNet, ReconNetV2
from trainer import FixedClassificationTrainer, AdaptiveClassificationTrainer, AnnealingClassificationTrainer, \
    ReconTrainer
from config import get_config
from dmd import FixedDMDAperture, FixedDMDSpatial
import torch

wandb.init("cdmd")

VERSION = 7


def main(config):
    torch.manual_seed(config.random_seed)
    torch.set_num_threads(1)
    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)
    train_loader, val_loader, image_size = dataloader.load_train_data(config.task, batch_size=config.batch_size,
                                                          resolution=config.resolution)
    if config.task.lower() == 'mnist' or config.task.lower() == 'cifar':
        adaptive = config.adaptive
        if adaptive:
            network_cls = AdaptiveDigitNet
            trainer_cls = AdaptiveClassificationTrainer
        else:
            network_cls = FixedDigitNet
            trainer_cls = FixedClassificationTrainer
    elif config.task.lower() == 'stl':
        adaptive = config.adaptive
        if adaptive:
            raise RuntimeError("Adaptive reconstruction is not supported")
        else:
            network_cls = ReconNetV2
            trainer_cls = ReconTrainer
    else:
        raise RuntimeError("Task name is invalid")

    if config.dmd_type.lower() == 'spatial':
        dmd_type = FixedDMDSpatial
    elif config.dmd_type.lower() == 'frequency':
        dmd_type = FixedDMDAperture
    else:
        raise RuntimeError("DMD type is invalid")

    network = network_cls(dmd_count=config.num_patterns, temperature=config.temp, hidden_size=config.hidden_size,
                          adaptive_multi=config.adaptive_multi, init_strategy=config.init_strategy,
                          resolution=config.resolution, noise=config.noise, dmd_type=dmd_type,
                          input_size=image_size)
    trainer = trainer_cls(network, train_loader, val_loader, init_lr=config.init_lr, epochs=config.epochs,
                          use_gpu=config.use_gpu)
    wandb.config.update(config)
    wandb.config.update({
        "version": VERSION
    })
    trainer.train()


if __name__ == "__main__":
    conf, unparsed = get_config()
    main(conf)
