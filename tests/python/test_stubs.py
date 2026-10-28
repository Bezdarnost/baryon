"""The type stub must describe the compiled module exactly.

The stub is what IDEs, type checkers and the documentation site read, while the runtime
docstrings come from the Rust source. These tests fail when the two drift apart.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import baryon
from baryon import _baryon

STUB = Path(baryon.__file__).with_name("_baryon.pyi")


def normalize(doc: str | None) -> str:
    return " ".join((doc or "").split())


def stub_docs() -> dict[str, str]:
    """Map ``name`` / ``Class.member`` to docstrings for every documented stub definition."""
    tree = ast.parse(STUB.read_text())
    docs: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.ClassDef):
            docs[node.name] = normalize(ast.get_docstring(node))
        if isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, ast.FunctionDef):
                    docs[f"{node.name}.{member.name}"] = normalize(ast.get_docstring(member))
    return docs


def runtime_docs() -> dict[str, str]:
    docs: dict[str, str] = {}
    for name, obj in vars(_baryon).items():
        if name.startswith("_"):
            continue
        if inspect.isclass(obj) or callable(obj):
            docs[name] = normalize(inspect.getdoc(obj))
        if inspect.isclass(obj) and obj.__module__ == "baryon":
            for member, value in vars(obj).items():
                if member.startswith("_") and member != "__array__":
                    continue
                docs[f"{name}.{member}"] = normalize(inspect.getdoc(value))
    return docs


STUB_DOCS = stub_docs()
RUNTIME_DOCS = runtime_docs()


def test_stub_and_runtime_define_the_same_names() -> None:
    assert set(STUB_DOCS) == set(RUNTIME_DOCS)


@pytest.mark.parametrize("name", sorted(RUNTIME_DOCS))
def test_docstrings_match(name: str) -> None:
    assert RUNTIME_DOCS[name], f"{name} has no runtime docstring"
    assert STUB_DOCS.get(name) == RUNTIME_DOCS[name]


def test_package_exports_everything_public() -> None:
    public = {name for name in vars(_baryon) if not name.startswith("_")}
    assert set(baryon.__all__) == public | {"__version__"}
