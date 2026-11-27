"""Assemble an already-built onedir app without deleting older builds."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    app = args.app_dir.resolve()
    if not (app / "Proter.exe").is_file() or not (app / "_internal").is_dir():
        parser.error("--app-dir must contain Proter.exe and _internal")
    if args.model.name != "yolov8n.pt" or not args.model.is_file():
        parser.error("--model must point to the selected yolov8n.pt")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"Proter-{args.version}-win64.zip"
    if archive.exists():
        parser.error(f"Refusing to overwrite {archive}")
    shutil.copy2(args.model, app / "yolov8n.pt")
    (app / "Calibration").mkdir(exist_ok=True)
    name = "manual_calibration_experimental_multidistance.json"
    shutil.copy2(ROOT / "Calibration" / name, app / "Calibration" / name)
    for name in ("QUICKSTART.md", "THIRD_PARTY_NOTICES.md"):
        shutil.copy2(ROOT / name, app / name)
    licenses = app / "licenses"
    licenses.mkdir(exist_ok=True)
    distributions = []
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        distributions.append({
            "name": name, "version": dist.version,
            "license": dist.metadata.get("License-Expression") or dist.metadata.get("License"),
        })
        for file in dist.files or []:
            path = Path(str(file))
            if any(token in path.name.lower() for token in ("license", "copying", "notice")):
                source = Path(dist.locate_file(file))
                if source.is_file() and source.suffix.lower() not in (".py", ".pyc", ".pyd"):
                    target = licenses / name / path.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists() and target.read_bytes() != source.read_bytes():
                        target = target.with_name(hashlib.sha256(str(file).encode()).hexdigest()[:8]
                                                  + "-" + path.name)
                    shutil.copy2(source, target)
    (licenses / "distributions.json").write_text(
        json.dumps(distributions, indent=2), encoding="utf-8")
    manifest = {}
    for file in sorted(app.rglob("*")):
        if file.is_file():
            with file.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            manifest[file.relative_to(app).as_posix()] = digest
    (app / "SHA256SUMS.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as zipped:
        for file in sorted(app.rglob("*")):
            if file.is_file():
                zipped.write(file, Path("Proter") / file.relative_to(app))
    with archive.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    archive.with_suffix(".zip.sha256").write_text(
        f"{checksum}  {archive.name}\n", encoding="ascii")
    print(archive)
    print("SHA256:", checksum)


if __name__ == "__main__":
    main()
