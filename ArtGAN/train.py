import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data_loader import get_dataloaders
from model import ConvRNNHybrid


def train_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        style_labels = labels['style'].to(device)
        artist_labels = labels['artist'].to(device)
        genre_labels = labels['genre'].to(device)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast():
            outputs = model(images)
            loss_style = criterion(outputs['style'], style_labels)
            loss_artist = criterion(outputs['artist'], artist_labels)
            loss_genre = criterion(outputs['genre'], genre_labels)
            total_loss = loss_style + loss_artist + loss_genre

        # Scaled backward pass
        scaler.scale(total_loss).backward()

        # Gradient clipping to prevent exploding gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        running_loss += total_loss.item()

    return running_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_style = 0
    correct_artist = 0
    correct_genre = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            style_labels = labels['style'].to(device)
            artist_labels = labels['artist'].to(device)
            genre_labels = labels['genre'].to(device)

            with autocast():
                outputs = model(images)
                loss_style = criterion(outputs['style'], style_labels)
                loss_artist = criterion(outputs['artist'], artist_labels)
                loss_genre = criterion(outputs['genre'], genre_labels)
                total_loss = loss_style + loss_artist + loss_genre

            running_loss += total_loss.item()

            _, pred_style = torch.max(outputs['style'], 1)
            _, pred_artist = torch.max(outputs['artist'], 1)
            _, pred_genre = torch.max(outputs['genre'], 1)

            correct_style += (pred_style == style_labels).sum().item()
            correct_artist += (pred_artist == artist_labels).sum().item()
            correct_genre += (pred_genre == genre_labels).sum().item()
            total += images.size(0)

    avg_loss = running_loss / len(dataloader)
    acc_style = correct_style / total
    acc_artist = correct_artist / total
    acc_genre = correct_genre / total

    return avg_loss, acc_style, acc_artist, acc_genre


def main():
    # --- Hyperparameters ---
    batch_size = 32
    num_epochs = 30
    backbone_lr = 1e-5       # Lower LR for pre-trained CNN layers
    head_lr = 1e-3            # Higher LR for new layers (RNN, heads)
    weight_decay = 1e-4       # L2 regularization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Checkpoint directory (saves to Drive if available) ---
    drive_path = '/content/drive/MyDrive/ArtGAN_Results'
    if os.path.exists('/content/drive/MyDrive'):
        ckpt_dir = os.path.join(drive_path, 'checkpoints')
    else:
        ckpt_dir = 'checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {ckpt_dir}")

    # --- 1. Load Data ---
    print("Loading data...")
    train_loader, val_loader = get_dataloaders(
        root_dir='.', batch_size=batch_size, num_workers=2, mock_images=False
    )

    # --- 2. Initialize Model ---
    print("Initializing model...")
    model = ConvRNNHybrid(
        num_styles=27, num_artists=23, num_genres=10,
        hidden_dim=512, num_rnn_layers=2, dropout=0.3, freeze_backbone=True
    )
    model = model.to(device)

    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total_params:,} ({100*trainable/total_params:.1f}%)")

    # --- 3. Differential Learning Rates ---
    # Separate backbone parameters from head/rnn parameters
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params, 'lr': head_lr}
    ], weight_decay=weight_decay)

    # --- 4. Learning Rate Scheduler ---
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # --- 5. Loss & Mixed Precision ---
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for better generalization
    scaler = GradScaler()

    # --- 6. Training Loop ---
    best_val_loss = float('inf')
    patience = 7
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        current_lrs = [pg['lr'] for pg in optimizer.param_groups]
        print(f"Learning Rates -> Backbone: {current_lrs[0]:.2e} | Head: {current_lrs[1]:.2e}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, acc_style, acc_artist, acc_genre = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy -> Style: {acc_style*100:.2f}% | Artist: {acc_artist*100:.2f}% | Genre: {acc_genre*100:.2f}%")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_path = os.path.join(ckpt_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'acc_style': acc_style,
                'acc_artist': acc_artist,
                'acc_genre': acc_genre,
            }, ckpt_path)
            print(f"*** Saved new best model to {ckpt_path} ***")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{patience})")

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping after {epoch+1} epochs.")
            break

    print("\n=== Training Complete ===")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Model saved at: {ckpt_dir}/best_model.pth")


if __name__ == '__main__':
    main()
