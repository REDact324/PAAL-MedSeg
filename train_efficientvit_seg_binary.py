"""Train an EfficientViT segmentation model on a binary segmentation dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch import amp
from torch.utils.data import DataLoader
from tqdm import tqdm

from binary_seg_utils import (
    BinarySegmentationDataset,
    BinarySegmentationMetrics,
    SegmentationEvalAugmentation,
    SegmentationTrainAugmentation,
    create_efficientvit_binary_segmentation_model,
    load_pretrained_segmentation_checkpoint,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

sys.path.append(ROOT_DIR)

from efficientvit.apps.utils import AverageMeter

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--train-image-dir", type=str, required=True, help="Directory with training images")
    parser.add_argument("--train-mask-dir", type=str, required=True, help="Directory with training masks")
    parser.add_argument("--val-image-dir", type=str, required=True, help="Directory with validation images")
    parser.add_argument("--val-mask-dir", type=str, required=True, help="Directory with validation masks")
    parser.add_argument("--model", type=str, default="l1", choices=["b0", "b1", "b2", "b3", "l1", "l2"])
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, nargs=2, default=(512, 512), metavar=("H", "W"))
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--save-frequency", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help=(
            "Optional checkpoint to initialize the model weights from. Shape-mismatched "
            "segmentation heads will be automatically skipped so multi-class weights can "
            "seed a binary fine-tuning run."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_transforms = SegmentationTrainAugmentation(resize=tuple(args.image_size), hflip_prob=args.hflip_prob)
    val_transforms = SegmentationEvalAugmentation(resize=tuple(args.image_size))

    train_dataset = BinarySegmentationDataset(
        args.train_image_dir,
        args.train_mask_dir,
        transforms=train_transforms,
    )
    val_dataset = BinarySegmentationDataset(
        args.val_image_dir,
        args.val_mask_dir,
        transforms=val_transforms,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def create_model_and_optimizer(args: argparse.Namespace) -> tuple[nn.Module, torch.optim.Optimizer]:
    model = create_efficientvit_binary_segmentation_model(args.model, num_classes=args.num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    return model, optimizer


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: amp.GradScaler | None,
) -> float:
    model.train()
    loss_meter = AverageMeter("train_loss")

    for batch in tqdm(dataloader, desc="Train", leave=False):
        images = batch["image"].to(device)
        targets = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with amp.autocast('cuda', enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, targets.long())

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[float, dict[str, float]]:
    model.eval()
    loss_meter = AverageMeter("val_loss")
    metrics = BinarySegmentationMetrics(num_classes=num_classes)

    for batch in tqdm(dataloader, desc="Eval", leave=False):
        images = batch["image"].to(device)
        targets = batch["mask"].to(device)

        logits = model(images)
        loss = criterion(logits, targets.long())
        loss_meter.update(loss.item(), images.size(0))

        predictions = torch.argmax(logits, dim=1)
        metrics.update(predictions.cpu(), targets.cpu())

    return loss_meter.avg, metrics.compute()


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    path: Path,
    args: argparse.Namespace,
) -> None:
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
        "args": vars(args),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, path: str) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint.get("epoch", 0)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_dataloaders(args)
    model, optimizer = create_model_and_optimizer(args)
    model = model.to(device)

    if args.pretrained and args.resume:
        raise ValueError("--pretrained and --resume cannot be used at the same time.")

    if args.pretrained:
        skipped, missing, unexpected = load_pretrained_segmentation_checkpoint(
            model,
            args.pretrained,
            num_classes=args.num_classes,
            strict=False,
        )
        if skipped:
            print(
                "Skipped loading the following parameters due to shape mismatch: "
                + ", ".join(skipped)
            )
        if missing:
            print("Missing keys after loading pretrained weights:", missing)
        if unexpected:
            print("Unexpected keys in pretrained checkpoint:", unexpected)

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    scaler = amp.GradScaler('cuda', enabled=args.amp)
    criterion = nn.CrossEntropyLoss()

    best_miou = -1.0
    history: list[dict[str, Any]] = []

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, args.num_classes)

        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"precision={val_metrics['precision']:.4f}, recall={val_metrics['recall']:.4f}, "
            f"mIoU={val_metrics['mean_iou']:.4f}, mDice={val_metrics['mean_dice']:.4f}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **val_metrics,
            }
        )

        checkpoint_dir = Path(args.checkpoint_dir)
        save_path = checkpoint_dir / "last.pt"
        save_checkpoint(model, optimizer, epoch + 1, val_metrics, save_path, args)

        if val_metrics["mean_iou"] > best_miou:
            best_miou = val_metrics["mean_iou"]
            save_checkpoint(model, optimizer, epoch + 1, val_metrics, checkpoint_dir / "best.pt", args)

        if (epoch + 1) % args.save_frequency == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                val_metrics,
                checkpoint_dir / f"epoch_{epoch + 1}.pt",
                args,
            )

    history_path = Path(args.checkpoint_dir) / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


if __name__ == "__main__":
    main()