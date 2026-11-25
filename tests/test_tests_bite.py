"""tools/tests_bite.py, run end to end on a throw-away repository with a known answer.

The repository has a base commit and a change. The change fixes `scale`, adds a module, and brings six tests:
one that fails on the base (it tests the change), one that passes there (it does not), one parametrized test
with a single failing set, one in a module the base cannot import, and two marked as characterization, one of
them without a reason. The tool has to name exactly the vacuous test and the unexplained mark.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

pytestmark = pytest.mark.characterization("exercises tools/tests_bite.py on a throw-away repository, not the package")

TOOL = Path(__file__).resolve().parents[1] / "tools" / "tests_bite.py"

NEW_TESTS = '''
import pytest
from microtopos import offset, scale


def test_bites():
    assert scale(2) == 6


def test_vacuous():
    assert offset(1) == 2


@pytest.mark.parametrize("x, want", [(0, 0), (1, 3)])
def test_one_set_bites(x, want):
    assert scale(x) == want


@pytest.mark.characterization("offset was already right; this pins it")
def test_marked():
    assert offset(0) == 1


@pytest.mark.characterization
def test_marked_without_reason():
    assert offset(0) == 1
'''


def _git(repo: Path, *args: str) -> None:
    quiet = ["-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false",
             "-c", "core.autocrlf=false"]
    subprocess.run(["git", *quiet, *args], cwd=repo, check=True, capture_output=True)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(text).lstrip("\n"), encoding="utf-8")


def _run(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(TOOL), "--base", "base", "--timeout", "300"], cwd=repo,
                          capture_output=True, text=True, check=False)


@pytest.fixture
def repo(tmp_path):
    repo = tmp_path / "repo"
    _write(repo / "pyproject.toml", '''
        [tool.pytest.ini_options]
        markers = ["slow", "characterization(reason)"]
    ''')
    _write(repo / "microtopos" / "__init__.py", '''
        def scale(x):
            return 2 * x


        def offset(x):
            return x + 1
    ''')
    _write(repo / "tests" / "test_old.py", '''
        from microtopos import scale


        def test_untouched():
            assert scale(0) == 0
    ''')
    _git(repo, "init", "-q", "-b", "base")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    _git(repo, "checkout", "-q", "-b", "change")
    init = repo / "microtopos" / "__init__.py"
    init.write_text(init.read_text(encoding="utf-8").replace("2 * x", "3 * x"), encoding="utf-8")
    _write(repo / "microtopos" / "extra.py", "ANSWER = 42\n")
    _write(repo / "tests" / "test_new.py", NEW_TESTS)
    _write(repo / "tests" / "test_newmod.py", '''
        from microtopos.extra import ANSWER


        def test_new_module():
            assert ANSWER == 42
    ''')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "change")
    return repo


def test_names_the_vacuous_test_and_the_unexplained_mark(repo):
    p = _run(repo)
    errors = sorted(line.strip() for line in p.stdout.splitlines() if "ERROR" in line)
    assert errors == [
        "ERROR characterization mark without a reason: tests/test_new.py::test_marked_without_reason",
        "ERROR passes on the base without this branch's change: tests/test_new.py::test_vacuous",
    ], p.stdout + p.stderr
    assert p.returncode == 1
    assert "6 new or changed test function(s)" in p.stdout  # test_untouched is not among them
    assert "3 of 4 judged test(s) fail on the base; 1 pass there; 0 not judged; 1 marked characterization" in p.stdout
    assert "characterization: tests/test_new.py::test_marked -- offset was already right; this pins it" in p.stdout


def test_passes_once_every_new_test_bites_or_says_why(repo):
    new = repo / "tests" / "test_new.py"
    text = new.read_text(encoding="utf-8")
    text = text.replace("def test_vacuous():\n    assert offset(1) == 2", "def test_vacuous():\n    assert scale(1) == 3")
    text = text.replace("@pytest.mark.characterization\n", '@pytest.mark.characterization("pins offset")\n')
    new.write_text(text, encoding="utf-8")  # left uncommitted: the working tree is what gets judged
    p = _run(repo)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "4 of 4 judged test(s) fail on the base; 0 pass there; 0 not judged; 2 marked characterization" in p.stdout


def test_sees_a_test_file_that_is_not_committed_yet(repo):
    _write(repo / "tests" / "test_untracked.py", '''
        from microtopos import offset


        def test_also_vacuous():
            assert offset(2) == 3
    ''')
    p = _run(repo)
    assert "ERROR passes on the base without this branch's change: tests/test_untracked.py::test_also_vacuous" in p.stdout
    assert p.returncode == 1
