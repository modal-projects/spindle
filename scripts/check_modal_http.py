"""Enforce authenticated Modal Servers across every tracked Python file.

Run with Python's standard library only; no Modal credentials or imports needed.
Web Function decorators are forbidden, including aliased imports. Every Server
must explicitly specify unauthenticated=False as a literal, so dynamic options
cannot silently weaken authentication.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

WEB_FUNCTIONS = {
    "asgi_app",
    "wsgi_app",
    "fastapi_endpoint",
    "web_endpoint",
    "web_server",
}


def violations(source: str, filename: str) -> list[str]:
    tree = ast.parse(source, filename=filename)
    aliases = {
        alias.asname or alias.name: alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "modal"
        for alias in node.names
    }
    errors = []
    for node in ast.walk(tree):
        name = (
            node.attr
            if isinstance(node, ast.Attribute)
            else aliases.get(node.id)
            if isinstance(node, ast.Name)
            else None
        )
        if name in WEB_FUNCTIONS:
            errors.append(
                f"{filename}:{node.lineno}: Modal {name} is forbidden; "
                "use @app.server(unauthenticated=False)"
            )
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "server":
            continue
        auth = next(
            (kw.value for kw in node.keywords if kw.arg == "unauthenticated"), None
        )
        if not isinstance(auth, ast.Constant) or auth.value is not False:
            errors.append(
                f"{filename}:{node.lineno}: Server must explicitly set unauthenticated=False"
            )
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = (
        subprocess.check_output(["git", "ls-files", "-z", "--", "*.py"], cwd=root)
        .decode()
        .split("\0")
    )
    errors = []
    for path in filter(None, paths):
        try:
            errors.extend(violations((root / path).read_text(), path))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f"{path}: cannot audit Python source: {exc}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(
        "Modal HTTP audit passed: no Web Functions; all Servers require authentication."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
