"""The test suite's own rules, checked mechanically (the reasons are in AGENTS.md).

1. Every test checks something: an `assert`, `pytest.raises` / `warns` / `fail`, a `numpy.testing` call, or a
   helper in this directory that does one of those. A test that only calls code proves that it does not raise.
2. Tests do not skip themselves. `pytest.skip`, `importorskip`, `skipif` and `pytest.xfail()` appear only in
   `conftest.py`, whose helpers skip for an absent capability and whose reasons are allowlisted there.
3. An expected failure is strict and says why, so that fixing the defect turns the build red until the entry
   is removed.
4. The allowlisted skip reasons name a missing capability, never a wrong or unstable result.

These rules look at the test files, not at the package, so on any base branch they pass or fail exactly as they
do here: `tools/tests_bite.py` cannot judge them, which the module-level mark below states.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.characterization("checks the test files themselves; independent of the package under test")

TESTS = Path(__file__).parent
CHECK_CALLS = {"pytest.raises", "pytest.warns", "pytest.fail", "pytest.deprecated_call"}
SKIP_NAMES = {
    "pytest.skip", "pytest.importorskip", "pytest.xfail", "pytest.mark.skip", "pytest.mark.skipif",
    "unittest.skip", "unittest.skipIf", "unittest.skipUnless", "unittest.SkipTest",
}
NOT_A_CAPABILITY = re.compile(
    r"diverg|wrong|mismatch|differ|inaccura|tolerance|flak|fail|broken|unstable|slow|time.?out|todo|known|bug", re.IGNORECASE
)


def parse_tests(folder: Path = TESTS) -> dict[str, ast.Module]:
    return {p.stem: ast.parse(p.read_text(encoding="utf-8"), filename=str(p)) for p in sorted(folder.glob("*.py"))}


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def _functions(tree: ast.Module) -> dict[str, ast.AST]:
    """Every function and method of a module by bare name (methods are found through `self.name(...)`)."""
    return {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _imports(tree: ast.Module, modules: set[str]) -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    """(local name -> (test module, function), local alias -> test module) for imports from this directory."""
    names: dict[str, tuple[str, str]] = {}
    aliases: dict[str, str] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            src = (n.module or "").rsplit(".", 1)[-1]
            for a in n.names:
                if src in modules:
                    names[a.asname or a.name] = (src, a.name)
                elif a.name in modules:  # from tests import _gen
                    aliases[a.asname or a.name] = a.name
        elif isinstance(n, ast.Import):
            for a in n.names:
                stem = a.name.rsplit(".", 1)[-1]
                if stem in modules:
                    aliases[a.asname or stem] = stem
    return names, aliases


def _checks_directly(fn: ast.AST) -> bool:
    for n in ast.walk(fn):
        if isinstance(n, ast.Assert):
            return True
        if isinstance(n, ast.Raise) and n.exc is not None and "AssertionError" in ast.dump(n.exc):
            return True
        if isinstance(n, ast.Call):
            name = _dotted(n.func)
            if name in CHECK_CALLS or ".testing." in f".{name}" or name.rsplit(".", 1)[-1].startswith("assert_"):
                return True
    return False


def unchecked_tests(mods: dict[str, ast.Module]) -> list[str]:
    """Tests that neither check anything themselves nor call a helper in this directory that does."""
    funcs = {m: _functions(t) for m, t in mods.items()}
    imports = {m: _imports(t, set(mods)) for m, t in mods.items()}
    checking = {(m, f) for m, fs in funcs.items() for f, node in fs.items() if _checks_directly(node)}

    def callees(m: str, node: ast.AST):
        names, aliases = imports[m]
        for n in ast.walk(node):
            if not isinstance(n, ast.Call):
                continue
            target = _dotted(n.func)
            head, _, tail = target.rpartition(".")
            if not head and target in funcs[m]:
                yield m, target
            elif not head and target in names:
                yield names[target]
            elif head in ("self", "cls") and tail in funcs[m]:
                yield m, tail
            elif head in aliases:
                yield aliases[head], tail

    grew = True
    while grew:  # a helper that calls a checking helper checks too
        grew = False
        for m, fs in funcs.items():
            for f, node in fs.items():
                if (m, f) not in checking and any(c in checking for c in callees(m, node)):
                    checking.add((m, f))
                    grew = True
    return sorted(f"{m}.py::{f}" for m, fs in funcs.items() if m.startswith("test_")
                  for f in fs if f.startswith("test") and (m, f) not in checking)


def skip_sites(mods: dict[str, ast.Module]) -> list[str]:
    """Places outside conftest.py where a test can take itself out of the run."""
    out = set()
    for m, tree in mods.items():
        if m == "conftest":
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute) and _dotted(n) in SKIP_NAMES:
                out.add((m, n.lineno, _dotted(n)))
            elif isinstance(n, ast.ImportFrom) and n.module in ("pytest", "unittest"):
                out |= {(m, n.lineno, f"from {n.module} import {a.name}") for a in n.names
                        if f"{n.module}.{a.name}" in SKIP_NAMES}
    return [f"{m}.py:{line}: {what}" for m, line, what in sorted(out)]


def loose_xfails(mods: dict[str, ast.Module]) -> list[str]:
    """`pytest.mark.xfail` used without `strict=True` or without a reason."""
    out = []
    for m, tree in mods.items():
        calls = {id(n.func): n for n in ast.walk(tree) if isinstance(n, ast.Call)}
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Attribute) and _dotted(n) == "pytest.mark.xfail"):
                continue
            kw = {k.arg: k.value for k in calls[id(n)].keywords} if id(n) in calls else {}
            strict = isinstance(kw.get("strict"), ast.Constant) and kw["strict"].value is True
            reason = kw.get("reason")
            empty = reason is None or (isinstance(reason, ast.Constant) and not str(reason.value or "").strip())
            if not strict or empty:
                out.append((m, n.lineno))
    return [f"{m}.py:{line}: xfail needs strict=True and a reason" for m, line in sorted(out)]


def allowed_skip_reasons(mods: dict[str, ast.Module]) -> list[str]:
    for n in ast.walk(mods["conftest"]):
        if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "ALLOWED_SKIP_REASONS" for t in n.targets):
            return [e.value for e in n.value.elts]
    raise AssertionError("tests/conftest.py no longer defines ALLOWED_SKIP_REASONS")


def test_every_test_checks_something():
    assert unchecked_tests(parse_tests()) == []


def test_tests_do_not_skip_themselves():
    assert skip_sites(parse_tests()) == []


def test_expected_failures_are_strict_and_explained():
    assert loose_xfails(parse_tests()) == []


def test_allowed_skip_reasons_name_a_missing_capability():
    reasons = allowed_skip_reasons(parse_tests())
    assert reasons, "the allowlist is empty: the strict-skip policy would reject every capability skip"
    assert [r for r in reasons if NOT_A_CAPABILITY.search(r)] == []


def _mods(**sources: str) -> dict[str, ast.Module]:
    return {name: ast.parse(src) for name, src in sources.items()}


def test_the_rules_catch_what_they_are_for():
    """Negative controls: each rule flags a violation written for it and accepts the legitimate form."""
    mods = _mods(
        _helpers="def check(x):\n    assert x\n\ndef relay(x):\n    check(x)\n\ndef plain(x):\n    return x\n",
        test_a=(
            "import pytest\nimport numpy as np\nimport _helpers as h\nfrom _helpers import relay, plain\n"
            "def test_only_runs():\n    plain(1)\n"
            "def test_finite_only():\n    np.isfinite(plain(1))\n"
            "def test_asserts():\n    assert plain(1) == 1\n"
            "def test_raises():\n    with pytest.raises(ValueError):\n        int('x')\n"
            "def test_numpy():\n    np.testing.assert_allclose(1, 1)\n"
            "def test_through_helper():\n    relay(1)\n"
            "def test_through_alias():\n    h.check(1)\n"
            "class TestC:\n    def _verify(self):\n        assert True\n"
            "    def test_method(self):\n        self._verify()\n"
            "    def test_empty_method(self):\n        plain(2)\n"
        ),
    )
    assert unchecked_tests(mods) == ["test_a.py::test_empty_method", "test_a.py::test_finite_only",
                                     "test_a.py::test_only_runs"]

    mods = _mods(
        conftest="import pytest\ndef require_x():\n    pytest.skip('no X device')\n",
        test_b=(
            "import pytest\nfrom pytest import importorskip\n"
            "pg = pytest.importorskip('pyqtgraph')\n"
            "@pytest.mark.skipif(True, reason='r')\ndef test_one():\n    assert 1\n"
            "def test_two():\n    pytest.skip('diverges')\n"
            "def test_three():\n    pytest.xfail('later')\n"
            "P = [pytest.param(1, marks=pytest.mark.skip)]\n"
        ),
    )
    assert skip_sites(mods) == [
        "test_b.py:2: from pytest import importorskip", "test_b.py:3: pytest.importorskip",
        "test_b.py:4: pytest.mark.skipif", "test_b.py:8: pytest.skip", "test_b.py:10: pytest.xfail",
        "test_b.py:11: pytest.mark.skip",
    ]  # the conftest helper's own pytest.skip is not listed

    mods = _mods(test_c=(
        "import pytest\n"
        "@pytest.mark.xfail\ndef test_bare():\n    assert 0\n"
        "@pytest.mark.xfail(reason='known')\ndef test_not_strict():\n    assert 0\n"
        "@pytest.mark.xfail(strict=True)\ndef test_no_reason():\n    assert 0\n"
        "@pytest.mark.xfail(strict=True, reason='  ')\ndef test_blank_reason():\n    assert 0\n"
        "@pytest.mark.xfail(strict=True, reason='issue 12: sign of dy')\ndef test_ok():\n    assert 0\n"
        "def table(why):\n    return pytest.param(1, marks=pytest.mark.xfail(strict=True, reason=why))\n"
    ))
    assert [s.split(":")[1] for s in loose_xfails(mods)] == ["2", "5", "8", "11"]

    assert NOT_A_CAPABILITY.search("OpenCL device diverges from the CPU result")
    assert NOT_A_CAPABILITY.search("flaky on Windows")
    assert not NOT_A_CAPABILITY.search("^no OpenCL device$")
    assert not NOT_A_CAPABILITY.search("needs Xvfb and a window manager")
