import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from pathlib import Path  # <- use pathlib

# Import the U-Net model and dataset class
from unet import UNet
from prostate_dataset import ProstateNiftiDataset

def dice_score(pred, target, threshold=0.5):
    # Apply sigmoid if using BCEWithLogitsLoss
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6
    dice = 2.0 * intersection / union
    return dice.mean().item()

if __name__ == "__main__":
    # Base dir of this script: D:\Escritorio\V4c_Prostate_DWI
    BASE_DIR = Path(__file__).resolve().parent

    # Paths (no hard-coded absolute paths)
    IMAGE_DIR = BASE_DIR / "Dataset_DWI" / "prostate" / "train_2d"
    MASK_DIR = BASE_DIR / "Dataset_DWI" / "prostate" / "train_mask_2d"
    MODEL_SAVE_PATH = BASE_DIR / "Models" / "unet_prostate_DWI.pth"
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Hyperparameters
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 32
    EPOCHS = 32

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Dataset (cast to str if your dataset expects strings)
    dataset = ProstateNiftiDataset(str(IMAGE_DIR), str(MASK_DIR))

    # Split
    generator = torch.Generator().manual_seed(42)
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len], generator=generator)

    # Loaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model, opt, loss
    model = UNet(in_channels=1, num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # Train
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0.0

        for img, mask in tqdm(train_dataloader, leave=False):
            img, mask = img.to(device), mask.to(device)
            y_pred = model(img)
            optimizer.zero_grad()
            loss = criterion(y_pred, mask)
            train_running_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss = train_running_loss / max(len(train_dataloader), 1)

        # Validate
        model.eval()
        val_running_loss = 0.0
        val_dice_scores = []
        with torch.no_grad():
            for img, mask in val_dataloader:
                img, mask = img.to(device), mask.to(device)
                y_pred = model(img)
                loss = criterion(y_pred, mask)
                val_running_loss += loss.item()
                val_dice_scores.append(dice_score(y_pred, mask))

        val_loss = val_running_loss / max(len(val_dataloader), 1)
        mean_dice = sum(val_dice_scores) / max(len(val_dice_scores), 1)
        print(f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {mean_dice:.4f}")

    # Save
    torch.save(model.state_dict(), str(MODEL_SAVE_PATH))
    print(f"Model saved to: {MODEL_SAVE_PATH}")
