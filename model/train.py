
"""
Traffic Forecasting Model Training Script

A clean and readable training script for traffic forecasting models.

Usage:
    python train.py -d PEMS08 -m train
    python train.py -d PEMS08 -m test
"""

import argparse
import datetime
import glob
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
import torch.nn as nn
import yaml
from torchinfo import summary

sys.path.append("..")

from lib.data_prepare import get_dataloader_from_index_data
from lib.metrics import MAE_MAPE_RMSE
from lib.utils import (
    CustomJSONEncoder,
    MaskedHuberLoss,
    MaskedMAELoss,
    masked_mae_loss,
    print_log,
    seed_random,
    set_cpu_num,
)
from model.TrafficModel import TrafficModel

def save_checkpoint(model, optimizer, scheduler, epoch, train_losses, val_losses, 
                   best_val_loss, best_epoch, config, checkpoint_dir):
    """保存训练断点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'config': config
    }
    
    checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # 保存最新断点信息
    latest_info = {
        'latest_epoch': epoch,
        'checkpoint_path': str(checkpoint_path)
    }
    
    with open(Path(checkpoint_dir) / "latest_checkpoint.json", 'w') as f:
        json.dump(latest_info, f)
    
    print(f"断点已保存: {checkpoint_path}")
    return checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """加载训练断点"""
    print(f"正在加载断点: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    best_val_loss = checkpoint['best_val_loss']
    best_epoch = checkpoint['best_epoch']
    
    print(f"断点加载成功，将从第 {start_epoch} 轮开始继续训练")
    print(f"历史最佳验证损失: {best_val_loss:.5f} (第 {best_epoch + 1} 轮)")
    
    return start_epoch, train_losses, val_losses, best_val_loss, best_epoch

def find_latest_checkpoint(checkpoint_dir):
    """查找最新的断点文件"""
    checkpoint_dir = Path(checkpoint_dir)
    latest_file = checkpoint_dir / "latest_checkpoint.json"
    
    if latest_file.exists():
        with open(latest_file, 'r') as f:
            latest_info = json.load(f)
        
        checkpoint_path = Path(latest_info['checkpoint_path'])
        if checkpoint_path.exists():
            return str(checkpoint_path)
    
    # 如果没有latest文件，查找最新的checkpoint文件
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if checkpoint_files:
        # 按epoch数字排序，返回最新的
        latest_checkpoint = max(checkpoint_files, 
                              key=lambda x: int(x.stem.split('_')[-1]))
        return str(latest_checkpoint)
    
    return None

def cleanup_old_checkpoints(checkpoint_dir, keep_last_n=3):
    """清理旧的断点文件，只保留最近的N个"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    
    if len(checkpoint_files) > keep_last_n:
        # 按epoch数字排序
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        
        # 删除旧的文件
        for old_file in checkpoint_files[:-keep_last_n]:
            old_file.unlink()
            print(f"删除旧断点: {old_file.name}")

def train_model_with_checkpoints(
    model, train_loader, val_loader, test_loader, optimizer, scheduler,
    criterion, scaler, device, config, save_path, log_file,
    checkpoint_interval=5, resume_from_checkpoint=True
):
    """带断点保存功能的训练函数"""
    print_log("Starting training with checkpoint support...", log=log_file)
    
    # 设置断点目录
    checkpoint_dir = Path(save_path).parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # 初始化训练参数
    start_epoch = 0
    best_val_loss = float('inf')
    best_epoch = 0
    patience = 0
    train_losses = []
    val_losses = []
    
    max_epochs = config.get('max_epochs', 200)
    early_stop = config.get('early_stop', 10)
    clip_grad = config.get('clip_grad')
    
    # 尝试从断点恢复
    if resume_from_checkpoint:
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            try:
                start_epoch, train_losses, val_losses, best_val_loss, best_epoch = load_checkpoint(
                    latest_checkpoint, model, optimizer, scheduler, device
                )
                patience = start_epoch - best_epoch  # 重新计算patience
                print_log(f"从断点恢复训练，开始轮次: {start_epoch}", log=log_file)
            except Exception as e:
                print_log(f"断点加载失败: {str(e)}，从头开始训练", log=log_file)
                start_epoch = 0
    
    print_log(f"训练配置: max_epochs={max_epochs}, early_stop={early_stop}, checkpoint_interval={checkpoint_interval}", log=log_file)
    
    for epoch in range(start_epoch, max_epochs):
        epoch_start_time = time.time()
        
        # 训练一个epoch
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, 
            scaler, device, clip_grad
        )
        train_losses.append(train_loss)
        
        # 验证
        val_loss = evaluate_model(model, val_loader, masked_mae_loss, scaler, device)
        val_losses.append(val_loss)
        
        # 测试（用于监控）
        test_loss = evaluate_model(model, test_loader, masked_mae_loss, scaler, device)
        
        epoch_time = time.time() - epoch_start_time
        
        # 记录日志
        print_log(
            f"Epoch {epoch + 1:3d}/{max_epochs} | "
            f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
            f"Test: {test_loss:.5f} | Time: {epoch_time:.2f}s",
            log=log_file
        )
        
        # 早停和保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience = 0
            torch.save(model.state_dict(), save_path)
            print_log(f"保存最佳模型 (Val Loss: {val_loss:.5f})", log=log_file)
        else:
            patience += 1
        
        # 定期保存断点
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = save_checkpoint(
                model, optimizer, scheduler, epoch, train_losses, val_losses,
                best_val_loss, best_epoch, config, checkpoint_dir
            )
            print_log(f"断点已保存: {checkpoint_path.name}", log=log_file)
            
            # 清理旧断点
            cleanup_old_checkpoints(checkpoint_dir, keep_last_n=3)
        
        # 早停检查
        if patience >= early_stop:
            print_log(f"早停触发，在第 {epoch + 1} 轮停止训练", log=log_file)
            # 保存最终断点
            final_checkpoint = save_checkpoint(
                model, optimizer, scheduler, epoch, train_losses, val_losses,
                best_val_loss, best_epoch, config, checkpoint_dir
            )
            print_log(f"最终断点已保存: {final_checkpoint.name}", log=log_file)
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load(save_path))
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, config['dataset'])
    
    # 最终评估
    final_evaluation(model, train_loader, val_loader, scaler, device, 
                    best_epoch, train_losses, val_losses, log_file)
    
    return model

@torch.no_grad()
def evaluate_model(model, data_loader, criterion, scaler, device):
    """Evaluate model performance"""
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
    
    return total_loss / num_batches if num_batches > 0 else float('inf')


@torch.no_grad()
def get_predictions(model, data_loader, scaler, device):
    """Get model predictions"""
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
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward pass
        predictions = model(x_batch)
        predictions = scaler.inverse_transform(predictions)
        loss = criterion(predictions, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    scheduler.step()
    return total_loss / num_batches if num_batches > 0 else float('inf')


def train_model(model, train_loader, val_loader, test_loader, optimizer, scheduler, 
                criterion, scaler, device, config, save_path, log_file):
    """Complete training pipeline"""
    print_log("Starting training...", log=log_file)
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience = 0
    train_losses = []
    val_losses = []
    
    max_epochs = config.get('max_epochs', 200)
    early_stop = config.get('early_stop', 10)
    clip_grad = config.get('clip_grad')
    
    for epoch in range(max_epochs):
        start_time = time.time()
        # Training
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, 
            scaler, device, clip_grad
        )
        train_losses.append(train_loss)
        
        # Validation
        val_loss = evaluate_model(model, val_loader, masked_mae_loss, scaler, device)
        val_losses.append(val_loss)
        
        # Testing (for monitoring)
        test_loss = evaluate_model(model, test_loader, masked_mae_loss, scaler, device)
        
        epoch_time = time.time() - start_time
        
        # Logging
        print_log(
            f"Epoch {epoch + 1:3d}/{max_epochs} | "
            f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
            f"Test: {test_loss:.5f} | Time: {epoch_time:.2f}s",
            log=log_file
        )
        
        # Early stopping and save best model
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
    
    # Load best model
    model.load_state_dict(torch.load(save_path))
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, config['dataset'])
    
    # Final evaluation
    final_evaluation(model, train_loader, val_loader, scaler, device, 
                    best_epoch, train_losses, val_losses, log_file)
    
    return model


def plot_training_curves(train_losses, val_losses, dataset):
    """Plot training curves"""
    try:
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs('./Figure', exist_ok=True)
        plt.savefig(f'./Figure/{dataset}-{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Warning: Could not save training plot: {str(e)}")


def final_evaluation(model, train_loader, val_loader, scaler, device, 
                    best_epoch, train_losses, val_losses, log_file):
    """Final evaluation"""
    # Calculate final metrics
    train_true, train_pred = get_predictions(model, train_loader, scaler, device)
    val_true, val_pred = get_predictions(model, val_loader, scaler, device)
    
    train_mae, train_mape, train_rmse = MAE_MAPE_RMSE(train_true, train_pred)
    val_mae, val_mape, val_rmse = MAE_MAPE_RMSE(val_true, val_pred)
    
    # Log results
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
    """Test model"""
    print_log("\nStarting model testing...", log=log_file)
    
    start_time = time.time()
    true_values, predictions = get_predictions(model, test_loader, scaler, device)
    inference_time = time.time() - start_time
    
    # Save predictions
    np.savez("test_value.npz", y_true=true_values, y_pred=predictions)
    print_log("Test results saved to: test_value.npz", log=log_file)
    
    # Overall metrics
    mae_all, mape_all, rmse_all = MAE_MAPE_RMSE(true_values, predictions)
    print_log(f"Overall test metrics - RMSE: {rmse_all:.5f}, MAE: {mae_all:.5f}, MAPE: {mape_all:.5f}%", log=log_file)
    
    # Per-step metrics
    num_steps = predictions.shape[1]
    print_log("\nPer-step test metrics:", log=log_file)
    for step in range(num_steps):
        step_mae, step_mape, step_rmse = MAE_MAPE_RMSE(
            true_values[:, step, :], predictions[:, step, :]
        )
        print_log(
            f"  Step {step + 1:2d}: RMSE={step_rmse:.5f}, MAE={step_mae:.5f}, MAPE={step_mape:.5f}%",
            log=log_file
        )
    
    print_log(f"\nInference time: {inference_time:.2f} seconds", log=log_file)
    print_log(f"Average time per sample: {inference_time / len(test_loader.dataset):.4f} seconds", log=log_file)


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Traffic Forecasting Model Training")
    parser.add_argument("-d", "--dataset", type=str, default="pems08", help="Dataset name")
    parser.add_argument("-g", "--gpu_num", type=int, default=0, help="GPU number")
    parser.add_argument("-m", "--mode", type=str, default="train", choices=["train", "test"], help="Running mode")
    parser.add_argument("-s", "--shift", action="store_true", help="Apply data shifting")
    
    args = parser.parse_args()
    
    # Setup environment
    seed_random(9)
    set_cpu_num(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    dataset = args.dataset.upper()
    data_path = f"../data/{dataset}"
    
    # Load configuration
    with open("TrafficModel.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)[dataset]
    config['dataset'] = dataset  # Add dataset name to config
    
    # Setup logging
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = Path(f"../logs/{dataset}")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(log_dir / f"TrafficModel-{dataset}-{timestamp}.log", "w", encoding="utf-8")
    
    print_log(f"Training session started: {timestamp}", log=log_file)
    print_log(f"Dataset: {dataset}", log=log_file)
    print_log(f"Device: {device}", log=log_file)
    print_log("-" * 50, log=log_file)
    
    try:
        # Load data
        print_log(f"Loading data from {data_path}...", log=log_file)
        train_loader, val_loader, test_loader, scaler, adj_matrices = get_dataloader_from_index_data(
            data_path,
            tod=config.get("time_of_day", True),
            dow=config.get("day_of_week", True),
            batch_size=config.get("batch_size", 32),
            log=log_file,
            train_ratio=config.get("train_size", 0.6),
            valid_ratio=config.get("val_size", 0.2),
            shift=args.shift,
        )
        
        # Prepare adjacency matrices
        supports = [torch.tensor(adj_matrix).to(device) for adj_matrix in adj_matrices]
        
        # Create model
        model = partial(TrafficModel, supports=None)
        model = model(**config["model_args"])
        model = model.to(device)
        
        # Print model info
        print_log("-" * 50, log=log_file)
        print_log("MODEL ARCHITECTURE", log=log_file)
        print_log("-" * 50, log=log_file)
        
        # Get input shape
        sample_batch = next(iter(train_loader))
        input_shape = [config.get("batch_size", 32)] + list(sample_batch[0].shape[1:])
        
        model_summary = summary(model, input_shape, verbose=0)
        print_log(str(model_summary), log=log_file)
        
        total_params = sum(p.numel() for p in model.parameters())
        print_log(f"Total parameters: {total_params:,}", log=log_file)
        print_log("-" * 50, log=log_file)
        
        # Setup save path
        save_dir = Path(f"../saved_models/TrafficModel-{dataset}")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if args.mode == "train":
            # Training mode
            save_path = save_dir / f"TrafficModel-{dataset}-{timestamp}.pt"
            shutil.copy2("TrafficModel.py", save_dir)
            
            # Setup training components
            criterion = MaskedMAELoss() # MaskedHuberLoss MaskedMAELoss
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
            
            # Train model
            model = train_model(
                model, train_loader, val_loader, test_loader,
                optimizer, scheduler, criterion, scaler, device,
                config, save_path, log_file
            )
            
            print_log(f"Model saved to: {save_path}", log=log_file)
            
        elif args.mode == "test":
            # Test mode - load latest model
            model_files = list(save_dir.glob(f"TrafficModel-{dataset}-*.pt"))
            if not model_files:
                raise FileNotFoundError(f"No saved models found in {save_dir}")
            
            latest_model = max(model_files, key=lambda x: x.stat().st_ctime)
            print_log(f"Loading model: {latest_model}", log=log_file)
            model.load_state_dict(torch.load(latest_model, map_location=device))
        
        # Test model
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
