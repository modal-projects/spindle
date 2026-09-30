"""Check built metadata; --for-pypi also enforces release prerequisites."""

import argparse
import ast
from email.parser import BytesParser
from pathlib import Path
import tarfile
import tomllib
import zipfile

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--for-pypi", action="store_true")
    parser.add_argument("--tag")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    errors = []

    version = None
    for statement in ast.parse((root / "src/spindle/__init__.py").read_text()).body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in statement.targets
        ):
            version = ast.literal_eval(statement.value)
    if version != project["version"]:
        errors.append("spindle.__version__ must match project.version")
    if args.tag is not None and args.tag != f"v{project['version']}":
        errors.append(f"Release tag must be v{project['version']}, got {args.tag!r}")

    wheels = list((root / "dist").glob("*.whl"))
    sdists = list((root / "dist").glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit(
            "Expected one wheel and one sdist in dist/; use a clean build directory"
        )
    with zipfile.ZipFile(wheels[0]) as archive:
        (metadata_path,) = (
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        wheel_metadata = BytesParser().parsebytes(archive.read(metadata_path))
        for source in (root / "src/spindle").rglob("*.py"):
            if source.relative_to(root / "src").as_posix() not in archive.namelist():
                errors.append(f"Wheel is missing {source.relative_to(root)}")
    with tarfile.open(sdists[0]) as archive:
        (metadata_path,) = (
            member
            for member in archive.getmembers()
            if member.name.count("/") == 1 and member.name.endswith("/PKG-INFO")
        )
        sdist_metadata = BytesParser().parsebytes(
            archive.extractfile(metadata_path).read()
        )

    for kind, metadata in [("wheel", wheel_metadata), ("sdist", sdist_metadata)]:
        for field, expected in [
            ("Name", "modal-spindle"),
            ("Version", project["version"]),
        ]:
            if metadata[field] != expected:
                errors.append(f"{kind}: {field} must be {expected!r}")
        if SpecifierSet(metadata["Requires-Python"]) != SpecifierSet(
            project["requires-python"]
        ):
            errors.append(f"{kind}: Requires-Python must match pyproject.toml")
        if not metadata["Summary"] or not metadata.get_payload().strip():
            errors.append(f"{kind}: missing description or README")
        if args.for_pypi:
            if not metadata["License-Expression"] or not metadata.get_all(
                "License-File"
            ):
                errors.append(
                    f"{kind}: choose a license and include its file before release"
                )
            for dependency in metadata.get_all("Requires-Dist", []):
                if Requirement(dependency).url:
                    errors.append(
                        f"{kind}: PyPI does not accept direct URL dependency: {dependency}"
                    )
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Validated modal-spindle {project['version']} wheel and sdist")


if __name__ == "__main__":
    main()
