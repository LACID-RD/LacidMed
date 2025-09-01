from __future__ import annotations
from pathlib import Path
import json
import nibabel as nib

# ---- Text fields commonly used in NIfTI headers ----
NIFTI_TEXT_FIELDS = ("descrip", "aux_file", "intent_name")

SIDECAR_PHI_KEYS = {
    "PatientName", "PatientID", "PatientBirthDate", "PatientSex", "PatientAge",
    "OtherPatientIDs", "OtherPatientNames", "PatientComments", "IssuerOfPatientID",
    "AccessionNumber", "StudyID", "ReferringPhysicianName", "RequestingPhysician",
    "PerformingPhysicianName", "OperatorsName", "InstitutionName", "InstitutionAddress",
    "StationName", "DeviceSerialNumber",
    "AcquisitionDate", "AcquisitionTime", "ContentDate", "ContentTime",
    "StudyDate", "StudyTime", "SeriesDate", "SeriesTime"
}

def _sidecar_json_for(nifti_path: Path) -> Path:
    name = nifti_path.name
    if name.endswith(".nii.gz"):
        return nifti_path.with_name(name[:-7] + ".json")
    elif name.endswith(".nii"):
        return nifti_path.with_suffix(".json")
    return nifti_path.with_suffix(".json")

def _tmp_with_same_ext(p: Path) -> Path:
    """Return a temp path that **keeps the original NIfTI extension at the end**."""
    suffixes = ''.join(p.suffixes)  # ".nii" or ".nii.gz"
    base = p.name[:-len(suffixes)] if suffixes else p.stem
    return p.with_name(f"{base}.tmp{suffixes}")

def _anon_nifti_header(img: nib.Nifti1Image, *, drop_orientation: bool = False, add_note: bool = True) -> None:
    hdr = img.header
    for f in NIFTI_TEXT_FIELDS:
        if f in hdr:
            try:
                hdr[f] = b""
            except Exception:
                pass

    # Remove any embedded extensions (can contain PHI)
    try:
        hdr.extensions.clear()
    except Exception:
        pass

    if drop_orientation:
        try:
            hdr["qform_code"] = 0
            hdr["sform_code"] = 0
            hdr["quatern_b"] = 0.0
            hdr["quatern_c"] = 0.0
            hdr["quatern_d"] = 0.0
            hdr["qoffset_x"] = 0.0
            hdr["qoffset_y"] = 0.0
            hdr["qoffset_z"] = 0.0
            hdr["srow_x"] = [1.0, 0.0, 0.0, 0.0]
            hdr["srow_y"] = [0.0, 1.0, 0.0, 0.0]
            hdr["srow_z"] = [0.0, 0.0, 1.0, 0.0]
        except Exception:
            pass

    if add_note and "descrip" in hdr:
        note = b"Anonymized: cleared text fields + extensions"
        hdr["descrip"] = note[:80].ljust(80, b"\x00")

def _anon_sidecar_json(json_path: Path) -> bool:
    if not json_path.exists():
        return False
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    changed = False
    for k in list(SIDECAR_PHI_KEYS):
        if k in data:
            del data[k]
            changed = True

    for k in ("SeriesDescription", "ProtocolName", "TaskName"):
        if k in data and isinstance(data[k], str):
            data[k] = ""

    if changed:
        tmp = json_path.with_name(json_path.stem + ".tmp.json")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(json_path)
    return changed

def anonymize_folder(images_dir: Path | str | None = None, *, recursive: bool = True, drop_orientation: bool = False):
    base_dir = Path(__file__).resolve().parent
    images_dir = Path(images_dir) if images_dir else (base_dir / "DATASET_DWI" / "Prostate" / "train_2d")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")

    pattern = "**/*.nii*" if recursive else "*.nii*"
    files = sorted(images_dir.glob(pattern))
    total = 0
    failed = 0
    sidecars = 0

    for nii_path in files:
        if not (nii_path.name.endswith(".nii") or nii_path.name.endswith(".nii.gz")):
            continue
        tmp_path = _tmp_with_same_ext(nii_path)
        try:
            img = nib.load(str(nii_path))
            _anon_nifti_header(img, drop_orientation=drop_orientation, add_note=True)

            # Save to a temp path that ends with .nii or .nii.gz
            nib.save(img, str(tmp_path))

            # Atomically replace the original
            tmp_path.replace(nii_path)
            total += 1

            sidecar = _sidecar_json_for(nii_path)
            if _anon_sidecar_json(sidecar):
                sidecars += 1

        except Exception as e:
            failed += 1
            try:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            print(f"Failed to anonymize {nii_path}: {e}")

    print(f"Anonymized {total} NIfTI file(s). Failed: {failed}. Sidecars cleaned: {sidecars}")

if __name__ == "__main__":
    # Example: anonymize current folder recursively
    anonymize_folder(Path.cwd(), recursive=True, drop_orientation=False)
