import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import nibabel as nib
from pathlib import Path

from unet import UNNet as UNet  # U-Net model

def single_image_inference(image_pth: str | Path,
                           og_pth: str | Path,
                           model_pth: str | Path,
                           device: str):
    """
    Perform inference on a single image and display the result.

    Args:
        image_pth: Path to the input NIfTI image (.nii / .nii.gz).
        og_pth: Path to the ground-truth mask NIfTI.
        model_pth: Path to the trained U-Net weights (.pth).
        device: 'cuda' or 'cpu'.
    """
    image_pth = Path(image_pth)
    og_pth = Path(og_pth)
    model_pth = Path(model_pth)

    # Load the trained model
    model = UNet(in_channels=1, num_classes=1).to(device)
    model.load_state_dict(torch.load(str(model_pth), map_location=torch.device(device)))
    model.eval()

    # Preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize((256, 256)),   # match model input size
        transforms.ToTensor()            # (H,W) -> (1,H,W) in [0,1]
    ])

    # ---- Load & preprocess input image ----
    nifti_img = nib.load(str(image_pth))
    img_array = nifti_img.get_fdata().astype("float32")

    # normalize safely
    mn, mx = float(img_array.min()), float(img_array.max())
    if mx > mn:
        img_array = (img_array - mn) / (mx - mn)
    else:
        img_array = img_array * 0.0  # constant image -> zeros

    # If 3D volume, pick middle slice (optional behavior)
    if img_array.ndim == 3:
        mid = img_array.shape[-1] // 2
        img_array = img_array[..., mid]

    img_pil = Image.fromarray((img_array * 255).astype("uint8"))
    img = transform(img_pil).float().to(device).unsqueeze(0)  # [1,1,H,W]

    # ---- Predict mask ----
    with torch.no_grad():
        logits = model(img)  # [1,1,H,W]
        pred_mask = torch.sigmoid(logits)
        pred_mask = (pred_mask > 0.5).float()

    # ---- Load & preprocess original mask ----
    nifti_mask = nib.load(str(og_pth))
    mask_array = nifti_mask.get_fdata().astype("float32")

    if mask_array.ndim == 3:
        mid = mask_array.shape[-1] // 2
        mask_array = mask_array[..., mid]

    mask_pil = Image.fromarray((mask_array > 0).astype("uint8") * 255)
    og_mask = transform(mask_pil).float().to(device).unsqueeze(0)  # [1,1,H,W]
    og_mask = (og_mask > 0).float()

    # ---- Prepare for visualization ----
    img_np = img.squeeze(0).cpu().permute(1, 2, 0).numpy()           # (H,W,1)
    og_mask_np = og_mask.squeeze(0).cpu().permute(1, 2, 0).numpy()   # (H,W,1)
    pred_mask_np = pred_mask.squeeze(0).cpu().permute(1, 2, 0).numpy()  # (H,W,1)

    img_masked_pred = img_np * pred_mask_np
    img_masked_og = img_np * og_mask_np

    # ---- Plot ----
    fig = plt.figure(figsize=(12, 4))

    fig.add_subplot(2, 3, 1)
    plt.imshow(img_np.squeeze(-1), cmap="gray")
    plt.title("Image")
    plt.axis("off")

    fig.add_subplot(2, 3, 2)
    plt.imshow(pred_mask_np.squeeze(-1), cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    fig.add_subplot(2, 3, 3)
    plt.imshow(og_mask_np.squeeze(-1), cmap="gray")
    plt.title("Original Mask")
    plt.axis("off")

    fig.add_subplot(2, 3, 5)
    plt.imshow(img_masked_pred.squeeze(-1), cmap="gray")
    plt.title("Image × Predicted Mask")
    plt.axis("off")

    fig.add_subplot(2, 3, 6)
    plt.imshow(img_masked_og.squeeze(-1), cmap="gray")
    plt.title("Image × Original Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Base directory of this script: D:\Escritorio\V4c_Prostate_DWI
    BASE_DIR = Path(__file__).resolve().parent

    # Paths using pathlib (no hard-coded absolute paths)
    #IF THE SEGMENTATION OF BRAIN IS DESIRED, "Prostate" should be changed to "Brain" (also needs to be changed in the model name in MODEL_PATH)
    SINGLE_IMG_PATH = BASE_DIR / "Dataset_DWI" / "Prostate" / "train_2d" / "patient5_7.nii"
    OG_IMG_PATH = BASE_DIR / "Dataset_DWI" / "Prostate" / "train_mask_2d" / "patient5_7.nii"
    MODEL_PATH = BASE_DIR / "Models" / "unet_prostate_DWI.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    single_image_inference(SINGLE_IMG_PATH, OG_IMG_PATH, MODEL_PATH, device)
