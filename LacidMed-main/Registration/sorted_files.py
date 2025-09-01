import re
from pathlib import Path

def _natural_key(p: Path):
    parts = re.split(r'(\d+)', p.name)
    return [int(s) if s.isdigit() else s.lower() for s in parts]

def main2(directory: str | Path | None = None):
    base_dir = Path(__file__).resolve().parent
    images_dir = Path(directory) if directory else (base_dir / "images")

    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")

    # Only .dcm files, naturally sorted (e.g., 1,2,10)
    directory_files = [str(p) for p in sorted(images_dir.glob("*.dcm"), key=_natural_key)]

    # Optional: warn about non-DICOM files present
    for p in sorted(images_dir.iterdir(), key=_natural_key):
        if p.is_file() and p.suffix.lower() != ".dcm":
            print(f"This file is not a DICOM file: {p}")

    return directory_files
