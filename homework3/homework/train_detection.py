# We have to create a training loop for the detection model
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import Classifier, load_model, save_model
from .metrics import AccuracyMetric, DetectionMetric, ConfusionMatrix
from .utils import load_data

def train_detection(
    exp_dir: str = "logs",
    model_name: str = "linear",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.built():
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
    model.train()

    train_data = load_data("classification_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("classification_data/val", shuffle=False)

    # create loss function and optimizer
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}
    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()

        model.train()

        for images, labels in train_data:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            acc = (outputs.max(1)[1] == labels).float().mean().item()
            metrics["train_acc"].append(acc) 
        model.eval()
        with torch.no_grad():
            for images, labels in val_data:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                acc = (outputs.max(1)[1] == labels).float().mean().item()
                metrics["val_acc"].append(acc)
        # log metrics to tensorboard
        logger.add_scalar("train_acc", np.mean(metrics["train_acc"]), global_step)
        logger.add_scalar("val_acc", np.mean(metrics["val_acc"]), global_step)
        print(
            f"Epoch {epoch+1}/{num_epoch}, "
            f"Train Acc: {np.mean(metrics['train_acc']):.4f}, "
            f"Val Acc: {np.mean(metrics['val_acc']):.4f}"
        )
        global_step += 1

        # save model checkpoint        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
        if epoch == num_epoch - 1:
            save_model(model, log_dir / f"{model_name}_epoch{epoch+1}.pt")
            logger.close()
        
    # save final model checkpoint at end of training
    save_model(model)

    # save a copy of the model weights to the homework directory for grading
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="linear")
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)

    args = parser.parse_args()

    train_detection(**vars(args))