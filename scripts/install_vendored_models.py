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


def _lightglue_package_data(repo_root: Path) -> list[str]:
    """Collect vendored LightGlue files relative to patcherbot package root."""
    package_root = repo_root / "patcherbot"
    lightglue_root = package_root / "deepLearning" / "cellModel" / "LightGlue"
    if not lightglue_root.exists():
        return []
    return [
        str(path.relative_to(package_root)).replace("\\", "/")
        for path in lightglue_root.rglob("*")
        if path.is_file()
    ]


def persist_lightglue_manifest(repo_root: Path) -> Path | None:
    """Write a manifest of LightGlue files for packaging or verification."""
    entries = _lightglue_package_data(repo_root)
    if not entries:
        return None
    manifest_path = repo_root / "patcherbot" / "deepLearning" / "cellModel" / "LightGlue" / "_package_data.txt"
    manifest_path.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return manifest_path


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
    manifest = persist_lightglue_manifest(repo_root)
    print(f"Wrote import helper: {pth_file}")
    if manifest:
        print(f"Wrote LightGlue package manifest: {manifest}")
    print("Done.")


if __name__ == "__main__":
    main()
