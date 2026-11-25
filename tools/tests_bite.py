"""Do the tests this branch adds or changes actually test its change?

A test that already passes on the base branch says nothing about the change it arrives with: it would have been
green without it. This tool finds every test function that is new or edited relative to the base, runs exactly
those tests against the BASE branch's `microtopos` package, and fails when one of them passes there.

    python tools/tests_bite.py --base origin/main          # what CI runs for a pull request
    python tools/tests_bite.py --base origin/main --list   # only list the changed tests

A test may legitimately pass on the base: a regression guard for behaviour that was already right, or a
characterisation of existing behaviour. Say so where a reviewer sees it (on the test, its class, or for a whole
module with `pytestmark = pytest.mark.characterization("...")`):

    @pytest.mark.characterization("guards the 4 px Gaussian cut-off rule that main already implements")

Limits (stated so nobody reads more into a green result than it holds):
* A test file that cannot even be imported on the base (it imports a module the branch adds) counts as failing
  on the base for every test in it. That is correct for "the feature did not exist", but it cannot tell a sharp
  test from a vacuous one. Mutation checks and review still decide that.
* Only the test function's own source is compared. A test whose helper or fixture changed is not re-examined.
* A test function with several parameter sets bites if at least one set fails on the base.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

PLUGIN = '''"""Loaded by tools/tests_bite.py: check which package is under test and record every outcome by node id."""
import json
import os
from pathlib import Path

import pytest

_results = {"tests": {}, "collect_errors": []}


def pytest_sessionstart(session):
    import microtopos

    base = Path(os.environ["TESTS_BITE_BASE"]).resolve()
    got = Path(microtopos.__file__).resolve()
    if base not in got.parents:
        pytest.exit(f"tests_bite: imported {got}, expected the base package under {base}", returncode=3)


def pytest_collection_modifyitems(config, items):
    """Whole files are collected (a file the base cannot import must not abort the run); keep the wanted tests."""
    want = set(json.loads(Path(os.environ["TESTS_BITE_WANT"]).read_text(encoding="utf-8")))
    keep = [i for i in items if i.nodeid.split("[", 1)[0] in want]  # node ids always use "/"
    drop = [i for i in items if i not in keep]
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep


def pytest_collectreport(report):
    if report.failed:
        _results["collect_errors"].append(report.nodeid)


def pytest_runtest_logreport(report):
    seen = _results["tests"].get(report.nodeid)
    if report.failed or hasattr(report, "wasxfail"):  # an expected failure is still a failure on the base
        _results["tests"][report.nodeid] = "failed"
    elif report.skipped and seen != "failed":
        _results["tests"][report.nodeid] = "skipped"
    elif report.when == "call" and report.passed and seen is None:
        _results["tests"][report.nodeid] = "passed"


def pytest_sessionfinish(session):
    Path(os.environ["TESTS_BITE_OUT"]).write_text(json.dumps(_results), encoding="utf-8")
'''


@dataclass
class Changed:
    file: str  # repository-relative, forward slashes
    qualname: str  # test_x or TestClass::test_x
    new: bool
    exempt_reason: str | None = None
    outcomes: list[str] = field(default_factory=list)  # per parameter set on the base: passed / failed / skipped

    @property
    def nodeid(self) -> str:
        return f"{self.file}::{self.qualname}"


def git(*args: str, cwd: Path, binary: bool = False):
    p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=False)
    if p.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{p.stderr.decode(errors='replace')}")
    return p.stdout if binary else p.stdout.decode(errors="replace")


def _mark_reason(mark: ast.AST) -> str | None:
    """The reason given to pytest.mark.characterization(...), '' when it has none, None for any other expression."""
    call = mark if isinstance(mark, ast.Call) else None
    target = call.func if call else mark
    if not (isinstance(target, ast.Attribute) and target.attr == "characterization"):
        return None
    if call and call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
        return call.args[0].value.strip()
    for kw in call.keywords if call else []:
        if kw.arg == "reason" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value.strip()
    return ""


def _characterization_reason(node: ast.AST) -> str | None:
    """The characterization reason on a function or class (decorator) or on a module (`pytestmark = ...`)."""
    marks = list(getattr(node, "decorator_list", []))
    for stmt in node.body if isinstance(node, ast.Module) else []:
        if isinstance(stmt, ast.Assign) and any(getattr(t, "id", "") == "pytestmark" for t in stmt.targets):
            marks += stmt.value.elts if isinstance(stmt.value, (ast.List, ast.Tuple)) else [stmt.value]
    return next((r for r in map(_mark_reason, marks) if r is not None), None)


def collect_tests(source: str) -> dict[str, tuple[str, str | None]]:
    """qualname -> (normalised source of the function, characterization reason) for every test in a module."""
    out: dict[str, tuple[str, str | None]] = {}
    tree = ast.parse(source)

    def visit(body, prefix: str, class_reason: str | None):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                reason = _characterization_reason(node)
                out[prefix + node.name] = (ast.dump(node), class_reason if reason is None else reason)
            elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                visit(node.body, f"{prefix}{node.name}::", _characterization_reason(node))

    visit(tree.body, "", _characterization_reason(tree))
    return out


def changed_tests(repo: Path, base: str) -> list[Changed]:
    names = git("diff", "--name-only", "--diff-filter=AM", base, "--", "tests", cwd=repo).split()
    names += git("ls-files", "--others", "--exclude-standard", "--", "tests", cwd=repo).split()  # not committed yet
    tests: list[Changed] = []
    for name in sorted({n for n in names if Path(n).name.startswith("test_") and n.endswith(".py")}):
        head = collect_tests((repo / name).read_text(encoding="utf-8"))
        shown = subprocess.run(["git", "show", f"{base}:{name}"], cwd=repo, capture_output=True, check=False)
        old = collect_tests(shown.stdout.decode("utf-8", errors="replace")) if shown.returncode == 0 else {}
        for qual, (dump, reason) in head.items():
            if qual not in old or old[qual][0] != dump:
                tests.append(Changed(name, qual, qual not in old, reason))
    return tests


def run_on_base(repo: Path, base: str, tests: list[Changed], timeout: int) -> None:
    """Fill Changed.outcomes by running the tests (from the working tree) against the base's package."""
    with tempfile.TemporaryDirectory(prefix="tests_bite_") as tmp:
        root = Path(tmp)
        with tarfile.open(fileobj=io.BytesIO(git("archive", "--format=tar", base, "microtopos", cwd=repo, binary=True))) as tar:
            tar.extractall(root, filter="data")
        (root / "tests_bite_plugin.py").write_text(PLUGIN, encoding="utf-8")
        out = root / "result.json"
        want = root / "want.json"
        want.write_text(json.dumps([t.nodeid for t in tests]), encoding="utf-8")
        path = os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        env = dict(os.environ, PYTHONPATH=path, TESTS_BITE_BASE=str(root), TESTS_BITE_OUT=str(out),
                   TESTS_BITE_WANT=str(want),
                   PYTHONDONTWRITEBYTECODE="1")
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
        # the skip policy must not turn "this cannot run on the base" into an error of this tool's own run
        env.pop("MICROTOPOS_STRICT_SKIPS", None)
        # one module that cannot be imported on the base must not stop the other modules from being judged
        cmd = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "tests_bite_plugin", "-q",
               "--continue-on-collection-errors",
               "-o", "addopts=", "-m", "slow or not slow", f"--rootdir={repo}", "-c", str(repo / "pyproject.toml"),
               *sorted({str(repo / t.file) for t in tests})]
        p = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, timeout=timeout, check=False)
        if p.returncode == 3 or not out.exists():
            raise SystemExit(f"tests_bite: the base run did not produce results (exit {p.returncode}).\n"
                             f"{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
        res = json.loads(out.read_text(encoding="utf-8"))
        broken = {n.replace("\\", "/") for n in res["collect_errors"]}
        for t in tests:
            t.outcomes = [o for nid, o in res["tests"].items() if nid.replace("\\", "/").split("[", 1)[0] == t.nodeid]
            if not t.outcomes and t.file in broken:
                t.outcomes = ["failed"]  # the module does not import on the base: the feature is not there


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="origin/main", help="branch or commit the change is measured against")
    ap.add_argument("--list", action="store_true", help="list the new or changed tests and stop")
    ap.add_argument("--timeout", type=int, default=2400, help="seconds allowed for the run on the base")
    a = ap.parse_args(argv)
    repo = Path(git("rev-parse", "--show-toplevel", cwd=Path.cwd()).strip())
    base = git("merge-base", a.base, "HEAD", cwd=repo).strip()
    tests = changed_tests(repo, base)
    print(f"tests_bite: {len(tests)} new or changed test function(s) relative to {a.base} ({base[:9]})")
    if a.list or not tests:
        for t in tests:
            print(f"  {'new    ' if t.new else 'changed'} {t.nodeid}")
        return 0
    no_reason = [t for t in tests if t.exempt_reason == ""]
    run_on_base(repo, base, [t for t in tests if t.exempt_reason is None], a.timeout)
    vacuous, unknown = [], []
    for t in tests:
        if t.exempt_reason is not None:
            continue
        if not t.outcomes:
            unknown.append(t)
        elif all(o == "passed" for o in t.outcomes):
            vacuous.append(t)
        elif all(o in ("passed", "skipped") for o in t.outcomes) and "skipped" in t.outcomes:
            unknown.append(t)
    for t in tests:
        if t.exempt_reason:
            print(f"  characterization: {t.nodeid} -- {t.exempt_reason}")
    for t in unknown:
        print(f"  NOT JUDGED (skipped or not collected on the base): {t.nodeid}")
    for t in no_reason:
        print(f"  ERROR characterization mark without a reason: {t.nodeid}")
    for t in vacuous:
        print(f"  ERROR passes on the base without this branch's change: {t.nodeid}")
    bad = len(vacuous) + len(no_reason)
    judged = sum(1 for t in tests if t.exempt_reason is None) - len(unknown)
    print(f"tests_bite: {judged - len(vacuous)} of {judged} judged test(s) fail on the base; {len(vacuous)} pass there; "
          f"{len(unknown)} not judged; {sum(1 for t in tests if t.exempt_reason)} marked characterization")
    if vacuous:
        print("A test that passes on the base does not test this change. Make it fail without the change, or mark it\n"
              "@pytest.mark.characterization(\"why it is expected to pass on the base\") if it guards existing behaviour.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
