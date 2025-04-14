import os
from os import PathLike
from pathlib import Path
from typing import cast

from google.api_core.exceptions import TooManyRequests
from google.cloud import storage
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm.contrib.concurrent import thread_map

from segmenter_api.domain.model.gcs import GCSPath
from segmenter_api.domain.repository.file import FileRepositoryInterface
from segmenter_api.settings import get_settings
from segmenter_api.utils.file import find_common_root
from segmenter_api.utils.logger import get_logger

logger = get_logger(__name__)

settings = get_settings()


class GCSRepository(FileRepositoryInterface):
    def __init__(self):
        self.client = storage.Client()
        self.bucket_name = settings.google_cloud_storage_bucket

    def download(
        self,
        source_paths: list[PathLike],
        destination_paths: list[Path],
        overwrite: bool = False,
    ) -> None:
        """
        GCSから複数のファイルを並列でダウンロード
        Args:
            source_paths: GCSPathのリスト
            destination_paths: ローカルファイルパスのリスト(source_pathsと同じ長さである必要があります)
            overwrite: 既存ファイルを上書きするかどうか(デフォルト: True)
        Raises:
            ValueError: source_pathsとdestination_pathsの長さが異なる場合
        """
        if len(source_paths) != len(destination_paths):
            error_msg = (
                f"source_paths (長さ: {len(source_paths)})と"
                f"destination_paths (長さ: {len(destination_paths)})の長さが一致しません"
            )
            raise ValueError(error_msg)

        # ディレクトリの中身を解析して、個々のファイルとして処理
        expanded_source_paths = []
        expanded_destination_paths = []
        for source_path, destination_path in zip(
            source_paths, destination_paths, strict=False
        ):
            gcs_path = GCSPath(bucket=self.bucket_name, blob_path=str(source_path))
            bucket = self.client.bucket(gcs_path.bucket)
            blobs = list(bucket.list_blobs(prefix=gcs_path.blob_path))

            if len(blobs) > 1 or (
                len(blobs) == 1 and blobs[0].name != gcs_path.blob_path
            ):
                # ディレクトリの場合
                for blob in blobs:
                    if blob.name == gcs_path.blob_path:  # ディレクトリ自体はスキップ
                        continue
                    relative_path = Path(blob.name).relative_to(gcs_path.blob_path)
                    expanded_source_paths.append(
                        GCSPath(bucket=self.bucket_name, blob_path=blob.name)
                    )
                    expanded_destination_paths.append(destination_path / relative_path)
            else:
                # ファイルの場合
                expanded_source_paths.append(gcs_path)
                expanded_destination_paths.append(destination_path)

        logger.info(
            f"""
            downloading {len(expanded_source_paths):,} files from
            bucket: {self.bucket_name}
            """
        )

        @retry(
            retry=retry_if_exception_type(TooManyRequests),
            wait=wait_exponential(multiplier=1, min=1, max=5),
            stop=stop_after_attempt(30),
            reraise=True,
        )
        def download_single(gcs_path: GCSPath, destination_path: Path) -> None:
            if not overwrite and destination_path.exists():
                return
            bucket = self.client.bucket(gcs_path.bucket)
            blob = bucket.blob(gcs_path.blob_path)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(destination_path))

        thread_map(
            download_single,
            expanded_source_paths,
            expanded_destination_paths,
            max_workers=min(len(expanded_source_paths), cast(int, os.cpu_count())),
        )

    def download_to_dir(
        self,
        source_paths: list[PathLike],
        destination_dir: Path,
        overwrite: bool = False,
    ) -> list[Path]:
        """
        GCSから複数のファイルを指定ディレクトリに並列でダウンロード
        Args:
            source_paths: GCSPathのリスト
            destination_dir: ダウンロード先のディレクトリパス
        """
        gcs_source_paths = [
            GCSPath(bucket=self.bucket_name, blob_path=str(path))
            for path in source_paths
        ]

        # GCSパスの共通ルートを見つける
        gcs_paths = [Path(gcs_path.blob_path) for gcs_path in gcs_source_paths]
        common_root = find_common_root(gcs_paths)

        destination_paths = [
            destination_dir / Path(gcs_path.blob_path).relative_to(common_root)
            for gcs_path in gcs_source_paths
        ]
        self.download(source_paths, destination_paths, overwrite)
        return destination_paths

    def upload(
        self,
        source_paths: list[Path],
        destination_paths: list[PathLike],
        overwrite: bool = False,
    ) -> None:
        """
        複数のファイルを並列でGCSにアップロード
        Args:
            source_paths: ローカルファイルパスのリスト
            destination_paths: GCSPathのリスト(source_pathsと同じ長さである必要があります)
            overwrite: 既存ファイルを上書きするかどうか(デフォルト: True)
        Raises:
            ValueError: source_pathsとdestination_pathsの長さが異なる場合
        """
        if len(source_paths) != len(destination_paths):
            error_msg = (
                f"source_paths (長さ: {len(source_paths)})と"
                f"destination_paths (長さ: {len(destination_paths)})の長さが一致しません"
            )
            raise ValueError(error_msg)

        gcs_destination_paths = [
            GCSPath(bucket=self.bucket_name, blob_path=str(path))
            for path in destination_paths
        ]

        @retry(
            retry=retry_if_exception_type(TooManyRequests),
            wait=wait_exponential(multiplier=1, min=1, max=5),
            stop=stop_after_attempt(30),
            reraise=True,
        )
        def upload_single(idx: int) -> None:
            local_path = source_paths[idx]
            gcs_path = gcs_destination_paths[idx]
            bucket = self.client.bucket(gcs_path.bucket)

            # ディレクトリかどうかを確認
            if local_path.is_dir():
                # ディレクトリの場合
                for file_path in local_path.rglob("*"):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(local_path)
                        blob_path = str(Path(gcs_path.blob_path) / relative_path)
                        blob = bucket.blob(blob_path)
                        if not overwrite and blob.exists():
                            continue
                        blob.upload_from_filename(str(file_path))
            else:
                # ファイルの場合
                blob = bucket.blob(gcs_path.blob_path)
                if not overwrite and blob.exists():
                    return
                blob.upload_from_filename(str(local_path))

        thread_map(
            upload_single,
            range(len(source_paths)),
            max_workers=min(len(source_paths), cast(int, os.cpu_count())),
        )

    def upload_to_dir(
        self,
        source_paths: list[Path],
        destination_dir: PathLike,
        overwrite: bool = False,
    ) -> list[PathLike]:
        """
        指定ディレクトリのファイルをGCSにアップロード
        """
        # ソースパスの共通ルートを見つける
        common_root = find_common_root(source_paths)

        destination_paths = [
            Path(destination_dir) / source_path.relative_to(common_root)
            for source_path in source_paths
        ]
        self.upload(
            source_paths=source_paths,
            destination_paths=list(destination_paths),
            overwrite=overwrite,
        )
        return list(destination_paths)
