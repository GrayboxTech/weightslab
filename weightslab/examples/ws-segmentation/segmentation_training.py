import logging
import os

from datetime import datetime

import hydra
import torch
import torch.backends.cudnn as cudnn

from omegaconf import OmegaConf

from datasets import Dataset
from evaluation import val_one
from model import netParams, SingleLiteNetPlus
from training import train
from training.callbacks import save_checkpoint, MultiMetricsEarlyStopping
from training.losses import SingleLoss
from training.lr_schedulers import poly_lr_scheduler
from utils import fix_randseed, log_metrics_single

try:
    import wandb
except:
    wandb = None

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train_net(config):
    LOGGER.info(
        f"##### CONFIGURATION #####\n{OmegaConf.to_yaml(config)}\n########################"
    )
    dict_configs = OmegaConf.to_container(config, resolve=True)
    """Train the neural network model with given arguments and hyperparameters"""
    # Setup wandb if enabled
    if config.logging.wandb and wandb:
        wandb.login()
        wandb.init(
            project="SingleLiteNetPlus",
            name=f"{config.logging.exp_name}_BDD_{config.dataset.images_folder}_{config.model.name}_{str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))}",
            config=dict_configs,
        )
        LOGGER.debug(f"WandB experiment initialized: {wandb.run.name}")

    fix_randseed(config.seed)
    cuda_available = torch.cuda.is_available()

    # Model initialization
    if config.model.version == "default":
        model = SingleLiteNetPlus(
            encoder_hp=config.model.encoder,
            caam_hp=config.model.caam,
            decoder_hp=config.model.decoder,
        )
    else:
        raise ValueError(f"Unknown model version: {config.model.version}")

    os.makedirs(config.savedir, exist_ok=True)
    LOGGER.debug("Loaded Model")
    trainLoader = torch.utils.data.DataLoader(
        Dataset(
            images_folder=config.dataset.images_folder,
            annotations_folder=config.dataset.annotations_folder,
            hyp=config.augmentation,
            model_name=config.model.name,
            split="train",
            lidar_img1=config.dataset.get("lidar_img1", None),
            lidar_img2=config.dataset.get("lidar_img2", None),
            lidar_img3=config.dataset.get("lidar_img3", None),
            idx_remapping=config.dataset.idx_remapping,
        ),
        batch_size=config.dataset.batch_size,
        shuffle=True,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
    )

    valLoader = torch.utils.data.DataLoader(
        Dataset(
            images_folder=config.dataset.images_folder,
            annotations_folder=config.dataset.annotations_folder,
            hyp=config.augmentation,
            model_name=config.model.name,
            split="val",
            lidar_img1=config.dataset.get("lidar_img1", None),
            lidar_img2=config.dataset.get("lidar_img2", None),
            lidar_img3=config.dataset.get("lidar_img3", None),
            idx_remapping=config.dataset.idx_remapping,
        ),
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
    )
    LOGGER.info("Loaded Data")

    if cuda_available:
        model = model.to(config.device)
        cudnn.benchmark = True

    LOGGER.info(f"Total network parameters: {netParams(model)}")

    criteria = SingleLoss(config.loss)
    start_epoch = 0
    lr = config.routine.lr
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(config.optimizer.momentum, 0.999),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    # Resume training from checkpoint
    if config.routine.resume and os.path.isfile(config.routine.resume):
        if config.routine.resume.endswith(".tar"):
            LOGGER.info(f"=> Loading checkpoint '{config.routine.resume}'")
            checkpoint = torch.load(config.routine.resume)
            start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            LOGGER.info(
                f"=> Loaded checkpoint '{config.routine.resume}' (epoch {checkpoint['epoch']})"
            )
        else:
            LOGGER.warning(f"=> No valid checkpoint found at '{config.routine.resume}'")

    scaler = (
        torch.amp.GradScaler(device=config.device)
        if config.routine.mixed_precision_training
        else None
    )

    if config.logging.wandb and wandb:
        # The log_freq should be at each epoch, so it should be equal to the number of batches
        # divided by the batch size.
        total_batches = len(trainLoader)
        log_freq = total_batches

        wandb.watch(model, log="all", log_freq=log_freq)
        LOGGER.info("WandB logging enabled, model parameters will be logged.")

    # Setup early stopping
    early_stopping = MultiMetricsEarlyStopping(
        patience=config.routine.early_stopping_patience,
        verbose=config.logging.verbose,
        modes={"val_da_miou": "max", "val_ll_iou": "max"},
    )

    LOGGER.info("Starting training...")
    for epoch in range(start_epoch, config.routine.max_epochs):
        model_file_name = os.path.join(config.savedir, f"model_{epoch}.pth")
        poly_lr_scheduler(
            config.routine.max_epochs, config.routine.lr, optimizer, epoch
        )
        for param_group in optimizer.param_groups:
            lr = param_group["lr"]
        LOGGER.info(f"Learning rate: {lr}")

        model.train()
        train(
            train_loader=trainLoader,
            model=model,
            criterion=criteria,
            optimizer=optimizer,
            epoch=epoch,
            scaler=scaler,
            device=config.device,
            mixed_precision_training=config.routine.mixed_precision_training,
            verbose=config.logging.verbose,
            logger=wandb if config.logging.wandb and wandb else None,
        )

        model.eval()
        metrics_computer = val_one(
            val_loader=valLoader,
            model=model,
            criteria=criteria,
            half=config.routine.mixed_precision_training,
            logger=wandb if config.logging.wandb and wandb else None,
            epoch=epoch,
            verbose=config.logging.verbose,
            device=config.device,
        )
        metrics = log_metrics_single(
            logger=wandb if config.logging.wandb and wandb else None,
            metrics=metrics_computer,
            epoch=epoch,
            class2idx=config.dataset.class2idx,
            idx_remapping=config.dataset.idx_remapping,
            split="val",
        )

        torch.save(model.state_dict(), model_file_name)

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr": lr,
            },
            os.path.join(config.savedir, "checkpoint.pth.tar"),
        )

        # Check for early stopping
        if early_stopping(
            {
                "val_da_miou": metrics["val/da_miou"],
                "val_ll_iou": metrics["val/ll_miou"],
            },
            epoch=epoch,
            checkpoint=model.state_dict(),
        ):
            LOGGER.info(f"Early stopping triggered. Best epoch: {early_stopping.best_epoch}")

            # Saving the best model
            best_model_path = os.path.join(config.savedir, "best_model.pth")
            torch.save(early_stopping.best_checkpoint, best_model_path)
            LOGGER.info(f"Best model saved at {best_model_path}")
            break

    if config.logging.wandb and wandb:
        wandb.finish()


if __name__ == "__main__":
    train_net()