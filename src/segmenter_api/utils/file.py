from pathlib import Path


def get_project_dir() -> Path:
    utils_dir = Path(__file__).parent
    pkg_dir = utils_dir.parent
    src_dir = pkg_dir.parent
    project_dir = src_dir.parent
    return project_dir
