import argparse
import logging
from pathlib import Path
from typing import List

import torch
import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf

from .checkpoint import save_checkpoint
from .eval import evaluate
from .lamar.dataset import LamarDataset
from .lamar.scene import SCENES as LAMAR_SCENES
from .lamar.scene import LamarScene
from .loss import APLoss
from .model import GCSA
from .msls.dataset import MslsDataset
from .msls.scene import SCENES_TRAIN as MSLS_SCENES_TRAIN
from .msls.scene import SCENES_VAL as MSLS_SCENES_VAL
from .msls.scene import MslsScene
from .utils import datasets_dir, outputs_dir


def create_dataset(conf: DictConfig, split: str, N: int, affinity: List[str]) -> torch.utils.data.dataset.Dataset:
    datasets = list()

    if conf.dataset == "msls":
        scenes = {
            "train": MSLS_SCENES_TRAIN,
            "val": MSLS_SCENES_VAL,
        }
        for name in tqdm.tqdm(scenes[split], desc="Create datasets"):
            if split == "train":
                subtask, labels = None, "fov"
            else:
                subtask, labels = "all", "dist"
            scene = MslsScene(name, datasets_dir() / conf.dataset, exclude_panos=True, subtask=subtask)
            desc_path = outputs_dir() / conf.dataset / name / conf.desc_fn
            datasets.append(MslsDataset(scene, desc_path, N, affinity, labels))
    elif conf.dataset == "lamar":
        for name in LAMAR_SCENES:
            scene = LamarScene(name, datasets_dir() / conf.dataset)
            output_dir = outputs_dir() / conf.dataset / name
            desc_path = output_dir / conf.desc_fn
            query_from_db = split == "train"
            radio_dropout_p = conf.radio_dropout if split == "train" else 0.0
            datasets.append(LamarDataset(scene, desc_path, N, query_from_db, affinity, radio_dropout_p=radio_dropout_p))
    else:
        raise ValueError(conf.dataset)

    return torch.utils.data.dataset.ConcatDataset(datasets)


def main(conf: DictConfig, name: str) -> None:
    # Create datasets
    N = max(conf.model.K, conf.model.L)
    train_set = create_dataset(conf.data, "train", N, list(conf.model.affinity))
    val_set = create_dataset(conf.data, "val", N, list(conf.model.affinity))
    logging.info(f"The training set has {len(train_set)} items and the validation set has {len(val_set)} items")

    # Select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Training on device {device}")

    # Create data loaders
    pin_memory = device == "cuda"
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=conf.data.batch_size,
        shuffle=True,
        num_workers=conf.data.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=conf.data.batch_size,
        shuffle=False,
        num_workers=conf.data.num_workers,
        pin_memory=pin_memory,
    )

    # Run evaluation without re-ranking
    logging.info("Results without re-ranking:")
    evaluate(val_loader, None, device)

    # Create model
    model_args = dict(
        input_dim=conf.model.input_dim,
        proj_dim=conf.model.proj_dim,
        output_dim=conf.model.output_dim,
        K=conf.model.K,
        L=conf.model.L,
        num_layers=conf.model.num_layers,
        num_heads=conf.model.num_heads,
        aff_dim=sum([conf.model.L if "db" in a else conf.model.L + 1 for a in conf.model.affinity]),
        input_dropout_p=conf.model.input_dropout,
        attn_dropout_p=conf.model.attn_dropout,
    )
    model = GCSA(**model_args).to(device)

    # Loss function
    loss_fnc = APLoss().to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.train.learning_rate, weight_decay=conf.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=conf.train.lr_step, gamma=conf.train.lr_gamma)
    scaler = torch.GradScaler(device)

    # Optionally load checkpoint to continue training from
    if conf.train.checkpoint:
        logging.info(f"Loading checkpoint from {conf.train.checkpoint}")
        checkpoint = torch.load(conf.train.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=False)

    # Optionally freeze the input projection
    if conf.train.freeze_input_proj:
        input_proj = model.input_proj
        logging.info(
            f"Freezing parameters of input projection ({input_proj.in_features} -> {input_proj.out_features} dim)"
        )
        for param in input_proj.parameters():
            param.requires_grad = False

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"The model has {num_params/1e6:.1f}M trainable parameters")

    checkpoint_dir = outputs_dir() / "training" / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    for epoch in range(1, conf.train.num_epochs + 1):
        train_loss = 0.0

        model.train()
        for desc, label, aff, _, _ in tqdm.tqdm(train_loader, smoothing=0.1, desc="Train"):
            desc, label, aff = desc.to(device), label.to(device), [a.to(device) for a in aff]

            optimizer.zero_grad()

            with torch.autocast(device):
                desc = model(desc, aff)
                sim = desc[:, 0].unsqueeze(1) @ desc.transpose(1, 2)
                loss = loss_fnc(sim.squeeze(1), label[:, : model.K + 1])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * desc.shape[0]

        train_loss /= len(train_loader.sampler)
        logging.info(f"Epoch: {epoch}/{conf.train.num_epochs} Avg train loss: {train_loss}")

        results = evaluate(val_loader, model, device)
        wandb.log(results, commit=False)
        wandb.log({"train_loss": train_loss, "learning_rate": scheduler.get_last_lr()[0]})

        checkpoint_path = checkpoint_dir / f"checkpoint_{epoch}.pth.tar"
        save_checkpoint(model, model_args, list(conf.model.affinity), checkpoint_path)

        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCSA training")
    parser.add_argument("--name", "-n", required=True, help="Name of the training run")
    parser.add_argument("--conf", "-c", type=Path, required=True, help="Path to config file")
    parser.add_argument("--wandb_project", "-wbprj", type=str, help="Weights & Biases project name")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("extra_args", nargs="*")
    args = parser.parse_args()

    conf = OmegaConf.load(args.conf)
    conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.extra_args))

    mode = "online" if args.wandb_project else "disabled"
    wandb.init(project=args.wandb_project, config=OmegaConf.to_container(conf), mode=mode, name=args.name)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s %(module)s %(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main(conf, args.name)
