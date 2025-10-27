"""Nox sessions."""

import nox
from nox_uv import session
from pathlib import Path
from textwrap import dedent

locations = ("tune", "tests", "noxfile.py")
nox.options.sessions = ("pre-commit", "tests")
python_versions = ["3.10", "3.11", "3.12", "3.13"]


def activate_virtualenv_in_precommit_hooks(session):
    """Activate virtualenv in hooks installed by pre-commit.
    This function patches git hooks installed by pre-commit to activate the
    session's virtual environment. This allows pre-commit to locate hooks in
    that environment when invoked from git.
    Args:
        session: The Session object.
    """
    if session.bin is None:
        return

    virtualenv = session.env.get("VIRTUAL_ENV")
    if virtualenv is None:
        return

    hookdir = Path(".git") / "hooks"
    if not hookdir.is_dir():
        return

    for hook in hookdir.iterdir():
        if hook.name.endswith(".sample") or not hook.is_file():
            continue

        text = hook.read_text()
        bindir = repr(session.bin)[1:-1]  # strip quotes
        if not (
            Path("A") == Path("a")
            and bindir.lower() in text.lower()
            or bindir in text
        ):
            continue

        lines = text.splitlines()
        if not (lines[0].startswith("#!") and "python" in lines[0].lower()):
            continue

        header = dedent(
            f"""\
            import os
            os.environ["VIRTUAL_ENV"] = {virtualenv!r}
            os.environ["PATH"] = os.pathsep.join((
                {session.bin!r},
                os.environ.get("PATH", ""),
            ))
            """
        )

        lines.insert(1, header)
        hook.write_text("\n".join(lines))


@session(python=python_versions, venv_backend="uv")
def tests(session):
    """Run the test suite."""
    session.install(".", "pytest")
    session.run("pytest", *session.posargs)


@session(python="3.13", venv_backend="uv")
def ruff(session):
    """Run Ruff formatter and linter."""
    args = session.posargs or locations
    session.install("ruff==0.14.2")
    session.run("ruff", "format", *args)
    session.run("ruff", "check", *args)


@session(name="pre-commit", python="3.13", venv_backend="uv")
def precommit(session):
    args = session.posargs or ["run", "--all-files", "--show-diff-on-failure"]
    session.install(
        "pre-commit",
        "ruff==0.14.2",
        "click",
    )
    session.run("pre-commit", *args)
    if args and args[0] == "install":
        activate_virtualenv_in_precommit_hooks(session)
