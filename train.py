import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import os
import json
import time
import random
import numpy as np
from datetime import datetime

# Import your modules
from dataset import RAVDESSDataset
from model import AudioOnlyModel, LateFusionModel, CrossAttentionModel

# --- CONFIGURATION ---
config = {
    "seed": 42,                  # <--- NEW: The magic number for reproducibility
    "batch_size": 8,
    "learning_rate": 1e-4,
    "epochs": 20,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_type": "cross_attention", 
    "data_path": "./ravdess_data",
    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
}

# Define filenames based on timestamp
log_filename = f"experiment_log_{config['model_type']}_{config['timestamp']}.json"
model_save_path = f"best_model_{config['model_type']}_{config['timestamp']}.pth"

def set_seed(seed):
    """
    Fixes the random seed for all libraries to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # These two lines ensure that the convolution algorithms used are deterministic
    # Note: This might slightly slow down training, but guarantees exact results
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random Seed set to: {seed}")

def save_metadata(log_data, filename):
    with open(filename, 'w') as f:
        json.dump(log_data, f, indent=4)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, leave=True)
    
    for batch in loop:
        audio = batch['audio'].to(device)
        video = batch['video'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        
        outputs = model(audio, video)
        
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loop.set_postfix(loss=loss.item(), acc=correct/total)

    return running_loss / len(loader), correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            audio = batch['audio'].to(device)
            video = batch['video'].to(device)
            labels = batch['label'].to(device)

            outputs = model(audio, video)
            
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            loss = criterion(logits, labels)

            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(loader), correct / total

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. SET SEED (Must be the very first thing!)
    set_seed(config["seed"])

    device = torch.device(config["device"])
    print(f"Starting Experiment: {config['model_type']}")
    print(f"Using Device: {device}")

    # 2. Initialize Log Dictionary
    experiment_log = {
        "config": config,
        "results": {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        },
        "best_epoch": 0,
        "best_val_acc": 0.0,
        "total_training_time": 0
    }

    # 3. Load Data
    # df = load_ravdess_data(config['data_path'])
    print("WARNING: Using Mock Data. Replace with real `load_ravdess_data` call.")
    df = pd.DataFrame([{
        'path': 'dummy.mp4', 'emotion_code': 1, 'actor': 1
    }] * 32) 

    train_df = df[df['actor'] <= 20].reset_index(drop=True)
    val_df = df[df['actor'] > 20].reset_index(drop=True)

    # Note: If shuffle=True, the random seed ensures the shuffle order is the same every time
    train_dataset = RAVDESSDataset(train_df, config['data_path'])
    val_dataset = RAVDESSDataset(val_df, config['data_path'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    # 4. Initialize Model
    if config['model_type'] == "audio_only":
        model = AudioOnlyModel(num_classes=8)
    elif config['model_type'] == "late_fusion":
        model = LateFusionModel(num_classes=8)
    elif config['model_type'] == "cross_attention":
        model = CrossAttentionModel(num_classes=8)
    
    model = model.to(device)

    # 5. Setup Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    start_time = time.time()

    # 6. Training Loop
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        experiment_log["results"]["train_loss"].append(train_loss)
        experiment_log["results"]["train_acc"].append(train_acc)
        experiment_log["results"]["val_loss"].append(val_loss)
        experiment_log["results"]["val_acc"].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > experiment_log["best_val_acc"]:
            experiment_log["best_val_acc"] = val_acc
            experiment_log["best_epoch"] = epoch + 1
            torch.save(model.state_dict(), model_save_path)
            print(f">>> New Best Model Saved to {model_save_path}!")

        save_metadata(experiment_log, log_filename)

    end_time = time.time()
    experiment_log["total_training_time"] = end_time - start_time
    save_metadata(experiment_log, log_filename)
    
    print("Training Complete.")
    print(f"Log saved to: {log_filename}")