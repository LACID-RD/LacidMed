from pathlib import Path
import pydicom
from pydicom.uid import generate_uid

# ---- what we blank/sanitize ----
PHI_FIELDS = [
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "PatientAge",
    "PatientAddress",
    "OtherPatientIDs",
    "OtherPatientNames",
    "PatientTelephoneNumbers",
    "PatientComments",
    "IssuerOfPatientID",
    "AccessionNumber",
    "StudyID",
    "ReferringPhysicianName",
    "RequestingPhysician",
    "PerformingPhysicianName",
    "OperatorsName",
    "InstitutionName",
    "InstitutionAddress",
    "StationName",
    "DeviceSerialNumber",
]

UID_FIELDS = [
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "FrameOfReferenceUID",
]

def _anon_ds(ds, uid_map):
    """
    Anonymize a pydicom Dataset in place.
    - Blanks PHI fields
    - Removes private tags
    - Replaces key UIDs with stable, new UIDs (consistent within run)
    """
    # Remove all private tags
    ds.remove_private_tags()

    # Blank PHI text fields (ignore if absent)
    for tag in PHI_FIELDS:
        if tag in ds:
            try:
                ds.data_element(tag).value = ""
            except Exception:
                pass

    # De-identification flags
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = "Basic de-id: PHI fields blanked; private tags removed; UIDs remapped"

    # Replace UIDs consistently
    for tag in UID_FIELDS:
        if tag in ds:
            old = str(ds.data_element(tag).value)
            new = uid_map.setdefault(old, generate_uid())
            ds.data_element(tag).value = new

    # Keep File Meta in sync with SOPInstanceUID if present
    if "SOPInstanceUID" in ds and hasattr(ds, "file_meta") and ds.file_meta is not None:
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID

    # Optional: if you know there are no burned-in pixels, set:
    # ds.BurnedInAnnotation = "NO"

def anonymize_folder(images_dir: Path | None = None):
    """
    Anonymize all .dcm files in the images folder (in place).
    The folder is resolved relative to this script by default.
    """
    base_dir = Path(__file__).resolve().parent
    images_dir = Path(images_dir) if images_dir else (base_dir / "images")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")

    uid_map: dict[str, str] = {}
    total = 0
    failed = 0

    for dcm_path in sorted(images_dir.glob("*.dcm")):
        tmp_path = dcm_path.with_suffix(dcm_path.suffix + ".tmp")
        try:
            ds = pydicom.dcmread(dcm_path)
            _anon_ds(ds, uid_map)
            # Save to temp, then atomically replace original
            ds.save_as(tmp_path, write_like_original=False)
            tmp_path.replace(dcm_path)
            total += 1
        except Exception as e:
            failed += 1
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            print(f"Failed to anonymize {dcm_path}: {e}")

    print(f"Anonymized {total} file(s). Failed: {failed}")

if __name__ == "__main__":
    anonymize_folder()  # defaults to <script_folder>/images