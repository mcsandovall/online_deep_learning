"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import PlannerMetric


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
    train_data = load_data("/content/online_deep_learning/homework4/drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("/content/online_deep_learning/homework4/drive_data/val", shuffle=False)

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

    loss_func = torch.nn.MSELoss()
    best_val_error = float("inf")

    # training loop
    for epoch in range(num_epoch):

        model.train()

        total_loss = 0.0

        for batch in train_data:
            batch = {k: v.to(device) for k, v in batch.items()}
            waypoints_gt = batch["waypoints"].to(device)
            track_left = batch["track_left"].to(device)
            track_right = batch["track_right"].to(device)

            # zero gradients for every batch
            optimizer.zero_grad()

            # predict waypoints and compute loss
            waypoints_pred = model(
              track_left=batch["track_left"],
              track_right=batch["track_right"],
            )

            # compute losses
            loss = loss_func(waypoints_pred, waypoints_gt)

            loss.backward()

            optimizer.step()

            # compute accumalative loss
            total_loss += loss.item()

        # step scheduler after each epoch
        scheduler.step()
        train_loss = total_loss / len(train_data)

        # validation
        model.eval()
        evaluator = PlannerMetric()

        with torch.no_grad():
            for batch in val_data:
                batch = {k: v.to(device) for k, v in batch.items()}
                waypoints_gt = batch["waypoints"].to(device)
                waypoints_mask = batch["waypoints_mask"].to(device)
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)

                waypoints_pred = model(
                  track_left=track_left,
                  track_right=track_right,
                )

                # get the segmentation predictions and update confusion matrix
                evaluator.add(waypoints_pred, waypoints_gt, waypoints_mask)
        
        # compute and log validation metrics
        val_metrics = evaluator.compute()

        logger.add_scalar("train_loss", train_loss, global_step=epoch)
        logger.add_scalar("val_loss", val_metrics["l1_error"] / val_metrics["num_samples"], global_step=epoch)
        logger.add_scalar("val_longitudinal_error", val_metrics["longitudinal_error"] / val_metrics["num_samples"], global_step=epoch)
        logger.add_scalar("val_lateral_error", val_metrics["lateral_error"] / val_metrics["num_samples"], global_step=epoch)

        # save best model
        if epoch == 0 or val_metrics["l1_error"] < best_val_error:
            best_val_error = val_metrics["l1_error"]
            save_model(model)
            print(f"Epoch {epoch}: New best model with L1 error {best_val_error:.4f}")

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="mlp_planner")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))


