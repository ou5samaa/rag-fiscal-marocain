"""Download the pinned E5 weight missing from the GitHub distribution."""
from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "data/search_reference/protocol.json").read_text(encoding="utf-8"))
TARGET = ROOT / PROTOCOL["model_dir"] / "model.safetensors"
EXPECTED = PROTOCOL["model_files"]["model.safetensors"]


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main() -> None:
    if TARGET.is_file() and digest(TARGET) == EXPECTED:
        print("Poids E5 déjà présents et vérifiés.")
        return
    if TARGET.exists():
        raise SystemExit("Poids E5 présents mais différents : ne pas les écraser automatiquement.")
    url = (f"https://huggingface.co/{PROTOCOL['model_id']}/resolve/"
           f"{PROTOCOL['model_revision']}/model.safetensors")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    pending = TARGET.with_suffix(".safetensors.download")
    if pending.exists():
        pending.unlink()
    try:
        with urllib.request.urlopen(url, timeout=120) as response, pending.open("wb") as output:
            shutil.copyfileobj(response, output, length=4 * 1024 * 1024)
        if digest(pending) != EXPECTED:
            raise ValueError("Empreinte SHA-256 du poids E5 incorrecte.")
        pending.replace(TARGET)
    finally:
        pending.unlink(missing_ok=True)
    print("Poids E5 téléchargés et vérifiés.")


if __name__ == "__main__":
    main()
