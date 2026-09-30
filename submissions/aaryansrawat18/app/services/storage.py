"""Where uploaded files live on disk.

The API and the worker share this folder. The file name is the document id,
so the queue only needs to carry that id.
"""

from pathlib import Path

from app.config import get_settings


def ensure_upload_directory() -> Path:
    """Create the shared upload folder if it is not there yet."""
    upload_directory = Path(get_settings().upload_dir)
    upload_directory.mkdir(parents=True, exist_ok=True)
    return upload_directory


def document_file_path(storage_key: str) -> Path:
    """Return the path for one document. `storage_key` is the document id."""
    return ensure_upload_directory() / storage_key
