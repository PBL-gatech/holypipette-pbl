"""Install the forked DL repos into patcherbot/deepLearning using --target."""

from __future__ import annotations

import subprocess
import sys
import sysconfig
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


def persist_vendor_paths(targets: list[Path]) -> Path:
    """Drop a .pth file into site-packages so vendored modules are importable."""
    site_packages = Path(sysconfig.get_paths()["purelib"])
    site_packages.mkdir(parents=True, exist_ok=True)
    pth_path = site_packages / "patcherbot_vendored.pth"
    resolved = [str(path.resolve()) for path in targets]
    pth_path.write_text("\n".join(resolved) + "\n", encoding="utf-8")
    return pth_path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    installed_targets = []
    for name, (rel_target, src) in VENDORED.items():
        target_dir = repo_root / rel_target
        install_target(name, target_dir, src)
        installed_targets.append(target_dir)
    pth_file = persist_vendor_paths(installed_targets)
    print(f"Wrote import helper: {pth_file}")
    print("Done.")


if __name__ == "__main__":
    main()
