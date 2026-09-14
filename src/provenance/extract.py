from __future__ import annotations

from io import BytesIO
from typing import Any

from PIL import Image


# Common EXIF tags we care about.
EXIF_TAGS = {
    271: "Make",
    272: "Model",
    305: "Software",
    306: "DateTime",
    274: "Orientation",
    34853: "GPSInfo",
}


def _safe_string(value: Any) -> str | None:
    """Convert metadata values into safe JSON-friendly strings."""
    if value is None:
        return None

    try:
        return str(value)
    except Exception:
        return None


def extract_exif(image_bytes: bytes) -> dict[str, Any]:
    """
    Extract useful EXIF metadata from the original image bytes.

    GPS coordinates are intentionally never returned.
    We only report whether GPS metadata exists.
    """

    result = {
        "present": False,
        "camera_make": None,
        "camera_model": None,
        "software": None,
        "datetime": None,
        "orientation": None,
        "gps_present": False,
    }

    try:
        image = Image.open(BytesIO(image_bytes))
        exif = image.getexif()

        if not exif:
            return result

        result["present"] = True

        for tag_id, value in exif.items():
            tag_name = EXIF_TAGS.get(tag_id)

            if tag_name == "Make":
                result["camera_make"] = _safe_string(value)

            elif tag_name == "Model":
                result["camera_model"] = _safe_string(value)

            elif tag_name == "Software":
                result["software"] = _safe_string(value)

            elif tag_name == "DateTime":
                result["datetime"] = _safe_string(value)

            elif tag_name == "Orientation":
                result["orientation"] = _safe_string(value)

            elif tag_name == "GPSInfo":
                # Do NOT expose GPS coordinates.
                result["gps_present"] = True

    except Exception:
        # Metadata extraction should never break image prediction.
        return result

    return result


def detect_c2pa(
    image_bytes: bytes,
    content_type: str | None = None,
) -> dict[str, Any]:
    """
    Read and validate embedded C2PA Content Credentials.

    Uses the installed c2pa-python Reader. No arbitrary byte/string
    matching is used as evidence of C2PA.
    """

    result = {
        "status": "unavailable",
        "verified": False,
        "title": None,
        "creator": None,
        "manifest_present": False,
        "validation_state": None,
        "validation_results": None,
    }

    try:
        import c2pa
    except ImportError:
        return result

    # Use the actual MIME type when available.
    # JPEG is the safest fallback for callers that don't provide one.
    mime_type = content_type or "image/jpeg"

    try:
        stream = BytesIO(image_bytes)

        with c2pa.Reader(mime_type, stream) as reader:

            # Check whether the image actually contains embedded
            # C2PA provenance data.
            result["manifest_present"] = reader.is_embedded()

            if not result["manifest_present"]:
                result["status"] = "not_detected"
                result["verified"] = False
                return result

            # Get the manifest JSON supplied by the C2PA reader.
            manifest_json = reader.json()

            # Check cryptographic validation.
            result["verified"] = bool(reader.is_valid())

            # Record the validation state as a string.
            try:
                validation_state = reader.get_validation_state()

                if validation_state is not None:
                    result["validation_state"] = str(
                        validation_state
                    )
            except Exception:
                pass

            # Keep validation details useful but bounded.
            try:
                validation_results = reader.get_validation_results()

                if validation_results is not None:
                    result["validation_results"] = str(
                        validation_results
                    )[:2000]
            except Exception:
                pass

            # Try to extract basic human-readable information
            # from the manifest JSON.
            if manifest_json:
                try:
                    import json

                    if isinstance(manifest_json, str):
                        manifest_data = json.loads(manifest_json)
                    else:
                        manifest_data = manifest_json

                    # C2PA manifests are structured JSON. We avoid
                    # assuming a single fixed schema and search only
                    # for common top-level descriptive fields.

                    if isinstance(manifest_data, dict):

                        title = manifest_data.get("title")
                        creator = manifest_data.get("creator")

                        if title:
                            result["title"] = str(title)

                        if creator:
                            result["creator"] = str(creator)

                except Exception:
                    pass

            if result["verified"]:
                result["status"] = "verified"
            else:
                result["status"] = "verification_failed"

    except Exception as exc:
        error_message = str(exc)

        # No embedded C2PA/JUMBF manifest is a normal result.
        # It does NOT mean that verification failed.
        if error_message.startswith("ManifestNotFound"):
            result["status"] = "not_detected"
            result["verified"] = False
            result["manifest_present"] = False
            result["validation_state"] = None
            result["validation_results"] = None
        else:
            # Something unexpected happened while inspecting C2PA.
            result["status"] = "verification_failed"
            result["verified"] = False
            result["manifest_present"] = False
            result["error"] = error_message[:500]

    return result


def analyze_provenance(
    image_bytes: bytes,
    filename: str | None = None,
    content_type: str | None = None,
) -> dict[str, Any]:
    """
    Analyze provenance-related metadata for an uploaded image.

    Returns:
        {
            "status": ...,
            "filename": ...,
            "content_type": ...,
            "c2pa": {...},
            "exif": {...},
            "summary": ...
        }
    """

    exif = extract_exif(image_bytes)
    c2pa_result = detect_c2pa(
        image_bytes,
        content_type,
    )

    if c2pa_result["status"] == "verified":
        summary = "Verified C2PA Content Credentials detected."

    elif c2pa_result["status"] == "verification_failed":
        summary = (
            "C2PA provenance was found, but its verification "
            "could not be completed."
        )

    elif exif["present"]:
        summary = (
            "EXIF metadata found. "
            "No C2PA Content Credentials detected."
        )

    else:
        summary = (
            "No EXIF metadata or C2PA Content Credentials "
            "were detected."
        )

    return {
        "status": "available",
        "filename": filename,
        "content_type": content_type,
        "c2pa": c2pa_result,
        "exif": exif,
        "summary": summary,
    }