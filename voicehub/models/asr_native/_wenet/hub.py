# Copyright (c) 2022 Mddct (hamddct@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified by VoiceHub in 2026: vendored and namespaced for inference-only use.

import shutil
import tarfile
import tempfile
from pathlib import Path

import requests
import tqdm

_MODEL_INDEX_URL = ("https://modelscope.cn/api/v1/datasets/wenet/"
                    "wenet_pretrained_models/oss/tree")


def _download_archive(url: str, destination: Path) -> Path:
    archive = destination / url.split("?", 1)[0].rsplit("/", 1)[-1]
    temporary = archive.with_suffix(f"{archive.suffix}.part")
    try:
        with requests.get(
                url,
                stream=True,
                timeout=(10, 120),
        ) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            with temporary.open("wb") as writer, tqdm.tqdm(
                    total=total or None,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=archive.name,
            ) as progress:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        writer.write(chunk)
                        progress.update(len(chunk))
        temporary.replace(archive)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return archive


def _extract_runtime_archive(archive: Path, destination: Path) -> None:
    """Extract regular child files while discarding archive path prefixes."""
    with tempfile.TemporaryDirectory(
            prefix=".wenet-extract-",
            dir=destination,
    ) as temporary_dir:
        temporary = Path(temporary_dir)
        with tarfile.open(archive, "r:*") as source:
            for member in source.getmembers():
                if not member.isfile() or "/" not in member.name:
                    continue
                source_file = source.extractfile(member)
                if source_file is None:
                    continue
                target = temporary / Path(member.name).name
                with source_file, target.open("wb") as writer:
                    shutil.copyfileobj(source_file, writer)

        required = {"final.zip", "units.txt"}
        available = {path.name for path in temporary.iterdir()}
        missing = sorted(required - available)
        if missing:
            raise RuntimeError(
                f"WeNet archive {archive.name!r} is missing required file(s): "
                f"{', '.join(missing)}.")
        for path in temporary.iterdir():
            path.replace(destination / path.name)


def download(
    url: str,
    dest: str | Path,
    only_child: bool = True,
) -> None:
    """Download and safely extract a WeNet TorchScript runtime archive."""
    if not only_child:
        raise ValueError("The vendored WeNet runtime only supports child-file extraction.")
    destination = Path(dest).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    archive = _download_archive(url, destination)
    try:
        _extract_runtime_archive(archive, destination)
    finally:
        archive.unlink(missing_ok=True)


class Hub:
    """Hub for WeNet pretrained TorchScript runtime models."""

    Assets = {
        "chinese": "wenetspeech_u2pp_conformer_libtorch.tar.gz",
        "english": "gigaspeech_u2pp_conformer_libtorch.tar.gz",
        "paraformer": "paraformer.tar.gz",
    }

    @staticmethod
    def get_model_by_lang(lang: str) -> str:
        if lang not in Hub.Assets:
            available = ", ".join(sorted(Hub.Assets))
            raise ValueError(f"Unsupported WeNet runtime {lang!r}. Available runtimes: "
                             f"{available}.")

        model = Hub.Assets[lang]
        model_dir = Path.home() / ".wenet" / lang
        model_dir.mkdir(parents=True, exist_ok=True)
        if {"final.zip", "units.txt"}.issubset({path.name for path in model_dir.iterdir()}):
            return str(model_dir)

        response = requests.get(
            _MODEL_INDEX_URL,
            timeout=(10, 30),
        )
        response.raise_for_status()
        payload = response.json()
        entries = payload.get("Data", ())
        model_info = next(
            (data for data in entries if data.get("Key") == model),
            None,
        )
        if model_info is None or not model_info.get("Url"):
            raise RuntimeError(f"ModelScope did not return the WeNet runtime asset {model!r}.")
        download(model_info["Url"], model_dir, only_child=True)
        return str(model_dir)
