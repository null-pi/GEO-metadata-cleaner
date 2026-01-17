import pathlib


def get_size_str(path: pathlib.Path) -> str:
    """Helper to get human-readable file size."""
    if not path.exists():
        return "-"

    size = path.stat().st_size
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"
