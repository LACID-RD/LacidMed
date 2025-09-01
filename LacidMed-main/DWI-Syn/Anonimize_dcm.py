from pathlib import Path
import pydicom
from pydicom.uid import generate_uid
from pydicom.tag import Tag

# ---- PHI fields to blank/sanitize (keep technical/acquisition fields) ----
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

# ---- UID fields to remap ----
UID_FIELDS = [
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "FrameOfReferenceUID",
]

# ---- Private tags that carry b-value (vendor-specific) we must PRESERVE ----
PHILIPS_B_TAG = Tag(0x2001, 0x1003)  # Philips: (2001,1003)
GE_B_TAG      = Tag(0x0043, 0x1039)  # GE:      (0043,1039)
KEEP_PRIVATE_TAGS = {PHILIPS_B_TAG, GE_B_TAG}

# ---- Standard DICOM b-value tag (will also be set for interoperability) ----
STD_B_TAG = Tag(0x0018, 0x9087)  # Diffusion b-value (FD)

def _extract_b_value(ds) -> int | float | None:
    """
    Extract the diffusion b-value from vendor-specific tags.
    Returns numeric b-value if found, else None.
    """
    # Philips private b-value
    if PHILIPS_B_TAG in ds:
        try:
            b = ds[PHILIPS_B_TAG].value
            # Some Philips datasets store as str or number; normalize to int if possible
            try:
                return int(b)
            except Exception:
                return float(b)
        except Exception:
            pass

    # GE private b-value (0043,1039) often encoded in a peculiar way
    if GE_B_TAG in ds:
        try:
            raw = ds[GE_B_TAG].value
            # Reproduce user's parsing logic for compatibility
            out2 = list(str(raw))
            try:
                if int(out2[1]) == 0:
                    return 0
                else:
                    # build thousands/hundreds/tens/ones from positions [7..10]
                    return int(out2[10]) + 10 * int(out2[9]) + 100 * int(out2[8]) + 1000 * int(out2[7])
            except Exception:
                # Fallback: take last 4 digits found in the string
                digits = [c for c in out2 if c.isdigit()]
                if digits:
                    return int("".join(digits[-4:]))
        except Exception:
            pass

    return None

def _remove_private_tags_except(ds, keep_tags: set[Tag], keep_groups: set[int] = None):
    """
    Remove all private tags EXCEPT:
      - those explicitly listed in keep_tags
      - private creator elements (0xgggg,0x00xx) for groups in keep_groups (to keep b-value creators)
    """
    if keep_groups is None:
        keep_groups = set()

    to_delete = []
    for elem in list(ds.iterall()):  # make a list to avoid modifying during iteration
        tag = elem.tag
        if not tag.is_private:
            continue

        # Keep explicit b-value private tags
        if tag in keep_tags:
            continue

        # Keep private creator elements in specified groups (e.g., 0x0043, 0x2001)
        # Private creators live at elements 0x0010–0x00FF
        if tag.group in keep_groups and 0x0010 <= tag.element <= 0x00FF:
            continue

        to_delete.append(tag)

    for tag in to_delete:
        del ds[tag]

def _anon_ds(ds, uid_map):
    """
    Anonymize a pydicom Dataset in place.
    - Blanks PHI fields
    - Removes private tags EXCEPT the vendor b-value tags (and their private creators)
    - Replaces key UIDs with stable, new UIDs (consistent within run)
    - Mirrors b-value into the standard DICOM tag (0018,9087) for interoperability
    """
    # --- 1) Capture b-value BEFORE touching private tags ---
    b_val = _extract_b_value(ds)

    # --- 2) Remove private tags selectively (preserve b-value tags & creators) ---
    _remove_private_tags_except(ds, KEEP_PRIVATE_TAGS, keep_groups={0x0043, 0x2001})

    # --- 3) Blank PHI text fields (ignore if absent) ---
    for tag_name in PHI_FIELDS:
        if tag_name in ds:
            try:
                ds.data_element(tag_name).value = ""
            except Exception:
                pass

    # --- 4) De-identification flags ---
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = (
        "Basic de-id: PHI fields blanked; private tags removed except vendor b-value; UIDs remapped"
    )

    # --- 5) Replace UIDs consistently ---
    for tag_name in UID_FIELDS:
        if tag_name in ds:
            old = str(ds.data_element(tag_name).value)
            new = uid_map.setdefault(old, generate_uid())
            ds.data_element(tag_name).value = new

    # Keep File Meta in sync with SOPInstanceUID if present
    if "SOPInstanceUID" in ds and hasattr(ds, "file_meta") and ds.file_meta is not None:
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID

    # --- 6) Also store b-value in STANDARD tag (0018,9087) if we have it ---
    if b_val is not None:
        try:
            # VR for (0018,9087) is FD (floating point). Cast to float for safety.
            ds.add_new(STD_B_TAG, "FD", float(b_val))
        except Exception:
            # If add_new fails for any reason, fall back silently.
            pass

    # Optional: if you know there are no burned-in pixels, set:
    # ds.BurnedInAnnotation = "NO"

def anonymize_folder(images_dir: Path | None = None):
    """
    Anonymize all .dcm files in the images folder (in place).
    Preserves vendor b-value tags (Philips 2001,1003; GE 0043,1039)
    and writes standard Diffusion b-value (0018,9087).
    """
    base_dir = Path(__file__).resolve().parent
    images_dir = Path(images_dir) if images_dir else (base_dir / "Ground_truth" / "patient1" / "Original")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")

    uid_map: dict[str, str] = {}
    total = 0
    failed = 0

    for dcm_path in sorted(images_dir.glob("*.dcm")):
        tmp_path = dcm_path.with_suffix(dcm_path.suffix + ".tmp")
        try:
            ds = pydicom.dcmread(str(dcm_path))
            _anon_ds(ds, uid_map)
            # Save to temp, then atomically replace original
            ds.save_as(str(tmp_path), write_like_original=False)
            tmp_path.replace(dcm_path)
            total += 1
        except Exception as e:
            failed += 1
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            print(f"Failed to anonymize {dcm_path}: {e}")

    print(f"Anonymized {total} file(s). Failed: {failed}")

if __name__ == "__main__":
    anonymize_folder()  # defaults to <script_folder>/Ground_truth/patient1/Original
