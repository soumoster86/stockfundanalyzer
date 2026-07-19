"""Static integrity checks on the live fundamentals app, src, and ui packages."""
from __future__ import annotations

import ast
import builtins
import importlib
import pathlib
import py_compile
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"
SRC = ROOT / "src"
UI = ROOT / "ui"

SRC_MODULES = [
    "auth",
    "build_labels",
    "data_quality",
    "enrich",
    "flag_lists",
    "governance",
    "institutional_scores",
    "model",
    "peers",
    "quality_score",
    "ranking",
    "red_flags",
    "sample_data",
    "schema",
    "screens",
    "watchlist",
]

UI_PY = [
    UI / "gauges.py",
    UI / "theme.py",
    UI / "tabs" / "report.py",
    UI / "tabs" / "ranking.py",
    UI / "tabs" / "compare.py",
    UI / "tabs" / "sector.py",
    UI / "tabs" / "train.py",
    UI / "tabs" / "watchlist_page.py",
    UI / "tabs" / "tutorial.py",
]


def _tree(path: pathlib.Path):
    return ast.parse(path.read_text(encoding="utf-8"))


def test_app_src_ui_compile():
    py_compile.compile(str(APP), doraise=True)
    for m in SRC_MODULES:
        py_compile.compile(str(SRC / f"{m}.py"), doraise=True)
    for path in UI_PY:
        py_compile.compile(str(path), doraise=True)


def test_app_imports_only_src_and_ui():
    """Live app must not depend on archived root modules."""
    tree = _tree(APP)
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod in {"auth", "data", "journal", "model"} and not mod.startswith(
                ("src", "ui")
            ):
                bad.append(mod)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name in {"auth", "data", "journal", "model"}:
                    bad.append(a.name)
    assert not bad, f"app.py imports archived root modules: {bad}"


def test_app_imports_resolve_from_src_and_ui():
    tree = _tree(APP)
    # Collect exports from src modules
    src_exports = {}
    for m in SRC_MODULES:
        names = set()
        for node in ast.walk(_tree(SRC / f"{m}.py")):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        names.add(t.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
        src_exports[m] = names

    # ui.tabs re-exports render_*
    ui_tabs_exports = {
        "render_report",
        "render_ranking",
        "render_compare",
        "render_sector",
        "render_train",
        "render_watchlist",
        "render_tutorial",
    }

    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        mod = node.module or ""
        if mod.startswith("src."):
            leaf = mod.split(".", 1)[1]
            if leaf not in src_exports:
                continue
            for a in node.names:
                if a.name == "*":
                    continue
                if a.name not in src_exports[leaf]:
                    bad.append(f"from {mod} import {a.name}")
        elif mod == "ui.tabs":
            for a in node.names:
                if a.name not in ui_tabs_exports:
                    bad.append(f"from {mod} import {a.name}")
    assert not bad, bad


def test_every_top_level_call_in_app_is_defined_or_imported():
    tree = _tree(APP)
    defined = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined.add(node.name)
            if isinstance(node, ast.FunctionDef):
                for a in node.args.args + node.args.kwonlyargs:
                    defined.add(a.arg)
        elif isinstance(node, ast.Import):
            defined.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            defined.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        defined.add(n.id)
        elif isinstance(node, (ast.For, ast.comprehension)):
            for n in ast.walk(node.target):
                if isinstance(n, ast.Name):
                    defined.add(n.id)
        elif isinstance(node, ast.withitem) and node.optional_vars:
            for n in ast.walk(node.optional_vars):
                if isinstance(n, ast.Name):
                    defined.add(n.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
    called = {
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    missing = sorted(called - defined)
    assert not missing, f"called but never defined/imported: {missing}"


def test_no_duplicate_static_widget_keys():
    keys = re.findall(r'key="([^"]+)"', APP.read_text(encoding="utf-8"))
    dups = sorted({k for k in keys if keys.count(k) > 1})
    assert not dups, f"duplicate widget keys: {dups}"


def test_third_party_imports_installable():
    local_roots = {"src", "app", "ui"}
    stdlib = set(sys.stdlib_module_names)
    optional = {"lightgbm", "xgboost", "yfinance"}  # optional local extras
    third_party = set()
    paths = [APP] + [SRC / f"{m}.py" for m in SRC_MODULES] + UI_PY
    for path in paths:
        for node in ast.walk(_tree(path)):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module.split(".")[0]]
            for m in mods:
                if m in stdlib or m in local_roots or m in optional:
                    continue
                if m.startswith("_"):
                    continue
                third_party.add(m)
    missing = []
    for pkg in sorted(third_party):
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    assert not missing, f"imported but not installed (fix requirements.txt): {missing}"
