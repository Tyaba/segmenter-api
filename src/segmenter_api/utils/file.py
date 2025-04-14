from pathlib import Path


def get_project_dir() -> Path:
    utils_dir = Path(__file__).parent
    pkg_dir = utils_dir.parent
    src_dir = pkg_dir.parent
    project_dir = src_dir.parent
    return project_dir


def find_common_root(paths: list[Path]) -> Path:
    """
    与えられたパスリストの共通ルートディレクトリを見つける
    Args:
        paths: パスのリスト
    Returns:
        共通のルートディレクトリのPath
    """
    if not paths:
        error_msg = "パスリストが空です"
        raise ValueError(error_msg)

    parts_list = [list(p.resolve().parts) for p in paths]
    min_length = min(len(parts) for parts in parts_list)

    common_parts = []
    for i in range(min_length):
        if len({parts[i] for parts in parts_list}) == 1:
            common_parts.append(parts_list[0][i])
        else:
            break

    if not common_parts:
        error_msg = "共通のルートディレクトリが見つかりません"
        raise ValueError(error_msg)

    common_part = Path(*common_parts)
    return common_part.relative_to(get_project_dir())
