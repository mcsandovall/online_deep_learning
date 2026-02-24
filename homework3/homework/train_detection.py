import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import DetectionMetric, ConfusionMatrix


def train(
    exp_dir: str = "logs",
    model_name: str = "detector",
    num_epoch: int = 50,
    lr: float = 1e-4,
    batch_size: int = 4,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)

    # load the dataset
    train_data = load_data("/content/online_deep_learning/homework3/drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("/content/online_deep_learning/homework3/drive_data/val", shuffle=False)

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epoch * len(train_data),
    )

    seg_loss_func = torch.nn.CrossEntropyLoss()
    depth_loss_func = torch.nn.L1Loss()

    best_iou = 0.0

    # training loop
    for epoch in range(num_epoch):

        model.train()

        total_loss = 0.0

        for batch in train_data:
            imgs = batch["image"].to(device)
            depth_gt = batch["depth"].to(device)
            seg_gt = batch["track"].to(device)

            # zero gradients for every batch
            optimizer.zero_grad()

            # model returns dict of losses during training
            seg_logits, depth_pred = model(imgs)

            # compute losses
            seg_loss = seg_loss_func(seg_logits, seg_gt)
            depth_loss = depth_loss_func(depth_pred, depth_gt)

            loss = seg_loss + depth_loss
            loss.backward()

            optimizer.step()

            # compute accumalative loss
            total_loss += loss.item()

        # step scheduler after each epoch
        scheduler.step()
        train_loss = total_loss / len(train_data)

        # validation
        model.eval()
        evaluator = ConfusionMatrix(num_classes=3)

        depth_mae = 0.0
        depth_lane_mae = 0.0
        total_pixels = 0
        total_lane_pixels = 0

        with torch.no_grad():
            for batch in val_data:
                imgs = batch["image"].to(device)
                depth_gt = batch["depth"].to(device)
                seg_gt = batch["track"].to(device)

                seg_logits, depth_pred = model(imgs)

                # get the segmentation predictions and update confusion matrix
                seg_pred = seg_logits.argmax(dim=1)
                evaluator.add(seg_pred.cpu(), seg_gt.cpu())

                # depth metrics
                abs_error = torch.abs(depth_pred - depth_gt)
                depth_mae += abs_error.sum().item()
                total_pixels += torch.numel(depth_gt)

                # lanel pixels only
                lane_mask = (seg_gt == 1) | (seg_gt == 2)

                depth_lane_mae += (abs_error * lane_mask).sum().item()
                total_lane_pixels += lane_mask.sum().item()
        
        # compute and log validation metrics
        val_metrics = evaluator.compute()
        mean_iou = val_metrics["iou"]

        logger.add_scalar("train_loss", train_loss, global_step=epoch)
        logger.add_scalar("val_loss", depth_mae / total_pixels, global_step=epoch)
        logger.add_scalar("val_lane_mae", depth_lane_mae / total_lane_pixels, global_step=epoch)
        logger.add_scalar("val_mean_iou", mean_iou, global_step=epoch)

        # save best model
        if mean_iou > best_iou:
            best_iou = mean_iou
            save_model(model)
            print(f"Epoch {epoch}: New best model with mean IoU {best_iou:.4f}")

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="detector")
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)

    args = parser.parse_args()

    train_detection(**vars(args))
