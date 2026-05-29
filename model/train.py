"""
Traffic Forecasting Model Training Script

Usage:
    python train.py -d PEMS08 -m train
    python train.py -d PEMS08 -m test
"""

import argparse
import datetime
import json
import os
import shutil
import sys
import time
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torchinfo import summary

sys.path.append("..")

from lib.data_prepare import get_dataloader_from_index_data
from lib.metrics import MAE_MAPE_RMSE
from lib.utils import MaskedMAELoss, masked_mae_loss, print_log, seed_random, set_cpu_num
from model.TrafficModel import TrafficModel


@torch.no_grad()
def evaluate_model(model, data_loader, criterion, scaler, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        predictions = model(x_batch)
        predictions = scaler.inverse_transform(predictions)
        loss = criterion(predictions, y_batch)
        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches if num_batches > 0 else float("inf")


@torch.no_grad()
def get_predictions(model, data_loader, scaler, device):
    model.eval()
    true_values = []
    predictions = []
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred_batch = model(x_batch)
        pred_batch = scaler.inverse_transform(pred_batch)
        predictions.append(pred_batch.cpu().numpy())
        true_values.append(y_batch.cpu().numpy())
    return np.vstack(true_values).squeeze(), np.vstack(predictions).squeeze()


def train_one_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, device, clip_grad=None):
    model.train()
    total_loss = 0.0
    num_batches = 0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        predictions = model(x_batch)
        predictions = scaler.inverse_transform(predictions)
        loss = criterion(predictions, y_batch)
        optimizer.zero_grad()
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    scheduler.step()
    return total_loss / num_batches if num_batches > 0 else float("inf")


def train_model(model, train_loader, val_loader, test_loader, optimizer, scheduler,
                criterion, scaler, device, config, save_path, log_file):
    print_log("Starting training...", log=log_file)
    best_val_loss = float("inf")
    best_epoch = 0
    patience = 0
    train_losses = []
    val_losses = []
    max_epochs = config.get("max_epochs", 200)
    early_stop = config.get("early_stop", 10)
    clip_grad = config.get("clip_grad")

    for epoch in range(max_epochs):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, device, clip_grad)
        train_losses.append(train_loss)
        val_loss = evaluate_model(model, val_loader, masked_mae_loss, scaler, device)
        val_losses.append(val_loss)
        test_loss = evaluate_model(model, test_loader, masked_mae_loss, scaler, device)
        epoch_time = time.time() - start_time
        print_log(
            f"Epoch {epoch + 1:3d}/{max_epochs} | Train: {train_loss:.5f} | "
            f"Val: {val_loss:.5f} | Test: {test_loss:.5f} | Time: {epoch_time:.2f}s",
            log=log_file,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience = 0
            torch.save(model.state_dict(), save_path)
            print_log(f"Saved best model (Val Loss: {val_loss:.5f})", log=log_file)
        else:
            patience += 1
        if patience >= early_stop:
            print_log(f"Early stopping triggered at epoch {epoch + 1}", log=log_file)
            break

    model.load_state_dict(torch.load(save_path, map_location=device))
    plot_training_curves(train_losses, val_losses, config["dataset"])
    final_evaluation(model, train_loader, val_loader, scaler, device, best_epoch, train_losses, val_losses, log_file)
    return model


def plot_training_curves(train_losses, val_losses, dataset):
    try:
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
        plt.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs("./Figure", exist_ok=True)
        plt.savefig(f"./Figure/{dataset}-{timestamp}.png", dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not save training plot: {str(e)}")


def final_evaluation(model, train_loader, val_loader, scaler, device, best_epoch, train_losses, val_losses, log_file):
    train_true, train_pred = get_predictions(model, train_loader, scaler, device)
    val_true, val_pred = get_predictions(model, val_loader, scaler, device)
    train_mae, train_mape, train_rmse = MAE_MAPE_RMSE(train_true, train_pred)
    val_mae, val_mape, val_rmse = MAE_MAPE_RMSE(val_true, val_pred)
    print_log("\n" + "=" * 50, log=log_file)
    print_log("TRAINING COMPLETED", log=log_file)
    print_log("=" * 50, log=log_file)
    print_log(f"Best epoch: {best_epoch + 1}", log=log_file)
    print_log(f"Best validation loss: {val_losses[best_epoch]:.5f}", log=log_file)
    print_log(f"Final training loss: {train_losses[best_epoch]:.5f}", log=log_file)
    print_log(f"Training metrics - RMSE: {train_rmse:.5f}, MAE: {train_mae:.5f}, MAPE: {train_mape:.5f}%", log=log_file)
    print_log(f"Validation metrics - RMSE: {val_rmse:.5f}, MAE: {val_mae:.5f}, MAPE: {val_mape:.5f}%", log=log_file)
    print_log("=" * 50, log=log_file)


def test_model(model, test_loader, scaler, device, log_file):
    print_log("\nStarting model testing...", log=log_file)
    start_time = time.time()
    true_values, predictions = get_predictions(model, test_loader, scaler, device)
    inference_time = time.time() - start_time
    np.savez("test_value.npz", y_true=true_values, y_pred=predictions)
    print_log("Test results saved to: test_value.npz", log=log_file)
    mae_all, mape_all, rmse_all = MAE_MAPE_RMSE(true_values, predictions)
    print_log(f"Overall test metrics - RMSE: {rmse_all:.5f}, MAE: {mae_all:.5f}, MAPE: {mape_all:.5f}%", log=log_file)
    num_steps = predictions.shape[1]
    print_log("\nPer-step test metrics:", log=log_file)
    for step in range(num_steps):
        step_mae, step_mape, step_rmse = MAE_MAPE_RMSE(true_values[:, step, :], predictions[:, step, :])
        print_log(f"  Step {step + 1:2d}: RMSE={step_rmse:.5f}, MAE={step_mae:.5f}, MAPE={step_mape:.5f}%", log=log_file)
    print_log(f"\nInference time: {inference_time:.2f} seconds", log=log_file)
    print_log(f"Average time per sample: {inference_time / len(test_loader.dataset):.4f} seconds", log=log_file)


def _prepare_supports(adj_matrices, device):
    if isinstance(adj_matrices, (list, tuple)):
        return [torch.as_tensor(adj, dtype=torch.float32, device=device) for adj in adj_matrices]
    return [torch.as_tensor(adj_matrices, dtype=torch.float32, device=device)]


def main():
    parser = argparse.ArgumentParser(description="Traffic Forecasting Model Training")
    parser.add_argument("-d", "--dataset", type=str, default="pems08", help="Dataset name")
    parser.add_argument("-g", "--gpu_num", type=int, default=0, help="GPU number")
    parser.add_argument("-m", "--mode", type=str, default="train", choices=["train", "test"], help="Running mode")
    parser.add_argument("-s", "--shift", action="store_true", help="Apply data shifting")
    args = parser.parse_args()

    seed_random(9)
    set_cpu_num(1)
    device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")
    dataset = args.dataset.upper()
    data_path = f"../data/{dataset}"

    with open("TrafficModel.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)[dataset]
    config["dataset"] = dataset

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = Path(f"../logs/{dataset}")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(log_dir / f"TrafficModel-{dataset}-{timestamp}.log", "w", encoding="utf-8")

    print_log(f"Training session started: {timestamp}", log=log_file)
    print_log(f"Dataset: {dataset}", log=log_file)
    print_log(f"Device: {device}", log=log_file)
    print_log("-" * 50, log=log_file)

    try:
        print_log(f"Loading data from {data_path}...", log=log_file)
        train_loader, val_loader, test_loader, scaler, adj_matrices = get_dataloader_from_index_data(
            data_path,
            tod=config.get("time_of_day", True),
            dow=config.get("day_of_week", True),
            batch_size=config.get("batch_size", 32),
            log=log_file,
            history_seq_length=config.get("input_steps", 12),
            pred_seq_length=config.get("output_steps", 12),
            train_ratio=config.get("train_size", 0.6),
            valid_ratio=config.get("val_size", 0.2),
            shift=args.shift,
        )
        supports = _prepare_supports(adj_matrices, device)

        model = partial(TrafficModel, supports=supports)
        model = model(**config["model_args"]).to(device)

        print_log("-" * 50, log=log_file)
        print_log("MODEL ARCHITECTURE", log=log_file)
        print_log("-" * 50, log=log_file)
        sample_batch = next(iter(train_loader))
        input_shape = [config.get("batch_size", 32)] + list(sample_batch[0].shape[1:])
        model_summary = summary(model, input_shape, verbose=0)
        print_log(str(model_summary), log=log_file)
        total_params = sum(p.numel() for p in model.parameters())
        print_log(f"Total parameters: {total_params:,}", log=log_file)
        print_log("-" * 50, log=log_file)

        save_dir = Path(f"../saved_models/TrafficModel-{dataset}")
        save_dir.mkdir(parents=True, exist_ok=True)
        if args.mode == "train":
            save_path = save_dir / f"TrafficModel-{dataset}-{timestamp}.pt"
            shutil.copy2("TrafficModel.py", save_dir)
            criterion = MaskedMAELoss()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.get("lr", 0.001),
                weight_decay=config.get("weight_decay", 0.0),
                eps=config.get("eps", 1e-8),
            )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=config.get("milestones", [30, 50, 70]),
                gamma=config.get("lr_decay_rate", 0.1),
            )
            print_log(f"Loss function: {criterion._get_name()}", log=log_file)
            print_log(f"Optimizer: Adam (lr={config.get('lr', 0.001)})", log=log_file)
            print_log("-" * 50, log=log_file)
            model = train_model(model, train_loader, val_loader, test_loader, optimizer, scheduler, criterion, scaler, device, config, save_path, log_file)
            print_log(f"Model saved to: {save_path}", log=log_file)
        elif args.mode == "test":
            model_files = list(save_dir.glob(f"TrafficModel-{dataset}-*.pt"))
            if not model_files:
                raise FileNotFoundError(f"No saved models found in {save_dir}")
            latest_model = max(model_files, key=lambda x: x.stat().st_ctime)
            print_log(f"Loading model: {latest_model}", log=log_file)
            model.load_state_dict(torch.load(latest_model, map_location=device))
        test_model(model, test_loader, scaler, device, log_file)
        print_log("Training session completed successfully!", log=log_file)
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        print(error_msg)
        print_log(error_msg, log=log_file)
        raise
    finally:
        log_file.close()


if __name__ == "__main__":
    main()
