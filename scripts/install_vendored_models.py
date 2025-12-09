"""Install the forked DL repos into patcherbot/deepLearning using --target."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

VENDORED = {
    "sam-2": (
        Path("patcherbot/deepLearning/cellModel/sam2"),
        "git+https://github.com/facebookresearch/sam2.git",
    ),
    "LightGlue": (
        Path("patcherbot/deepLearning/cellModel/LightGlue"),
        "git+https://github.com/PBL-gatech/LightGlue.git",
    ),
    "robomimic": (
        Path("patcherbot/deepLearning/patchModel/robomimic"),
        "git+https://github.com/PBL-gatech/robomimic.git",
    ),
}


def install_target(name: str, target: Path, source: str) -> None:
    target.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--no-deps",
        "--target",
        str(target),
        source,
    ]
    print(f"[{name}] installing into {target} ...")
    subprocess.check_call(cmd)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for name, (rel_target, src) in VENDORED.items():
        install_target(name, repo_root / rel_target, src)
    print("Done.")


if __name__ == "__main__":
    main()
