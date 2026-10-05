"""Tests for the holdout lock guards, the unlock log and outcome/verdict helpers (temp git repo)."""

from __future__ import annotations

import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.research.longshot_fade import config as C
from src.research.longshot_fade.unlock import (
    LockError,
    append_log,
    check_lock,
    holdout_logged,
    outcome_from_market,
    verdict,
)

UTC = timezone.utc


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture()
def repo(tmp_path: Path):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    (r / ".gitignore").write_text("data/\n")
    (r / "code.py").write_text("x = 1\n")
    _git(r, "add", ".")
    _git(r, "commit", "-q", "-m", "freeze")
    _git(r, "tag", "prereg-frozen")
    panel = r / "data" / "panel"
    f = panel / "split=holdA" / "quarter=2025Q4" / "part-0.parquet"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"holdout bytes")
    expected = {C.rel(f, r): C.sha256_file(f)}
    log = r / "data" / "HOLDOUT_LOG.md"
    log.write_text("# log\n")
    return {"root": r, "panel": panel, "file": f, "expected": expected, "log": log}


def _check(repo, holdout="A", tag="prereg-frozen", **kw):
    return check_lock(
        holdout,
        root=repo["root"],
        expected_tag=tag,
        expected_files=repo["expected"],
        panel_dir=repo["panel"],
        split="holdA",
        log_path=repo["log"],
        **kw,
    )


def test_all_checks_pass(repo):
    out = _check(repo)
    assert out["files_verified"] == 1
    assert out["head"] == _git(repo["root"], "rev-parse", "HEAD")


def test_refuses_when_head_is_not_tag(repo):
    (repo["root"] / "code.py").write_text("x = 2\n")
    _git(repo["root"], "commit", "-qam", "later")
    with pytest.raises(LockError, match="not the commit tagged"):
        _check(repo)


def test_refuses_missing_tag(repo):
    with pytest.raises(LockError, match="cannot resolve"):
        _check(repo, tag="no-such-tag")


def test_refuses_dirty_tree(repo):
    (repo["root"] / "code.py").write_text("x = 3\n")
    with pytest.raises(LockError, match="not clean"):
        _check(repo)


def test_refuses_untracked_file(repo):
    (repo["root"] / "new.txt").write_text("hi\n")
    with pytest.raises(LockError, match="not clean"):
        _check(repo)


def test_refuses_hash_mismatch(repo):
    repo["file"].write_bytes(b"tampered")
    with pytest.raises(LockError, match="SHA-256 mismatch"):
        _check(repo)


def test_refuses_extra_or_missing_file(repo):
    extra = repo["panel"] / "split=holdA" / "quarter=2025Q4" / "part-1.parquet"
    extra.write_bytes(b"x")
    with pytest.raises(LockError, match="file set differs"):
        _check(repo)
    extra.unlink()
    repo["file"].unlink()
    with pytest.raises(LockError, match="file set differs"):
        _check(repo)


def test_refuses_after_logged_unlock(repo):
    head = _git(repo["root"], "rev-parse", "HEAD")
    append_log(repo["log"], "A", head, "abc", datetime(2026, 10, 6, tzinfo=UTC))
    assert holdout_logged(repo["log"], "A")
    assert not holdout_logged(repo["log"], "B")
    text = repo["log"].read_text()
    assert "unlocked once" in text and head in text
    with pytest.raises(LockError, match="only run once"):
        _check(repo)


def test_holdout_b_wait(repo):
    now = datetime(2026, 10, 20, tzinfo=UTC)
    with pytest.raises(LockError, match="not allowed before"):
        _check(repo, holdout="B", pull_completed_at=now - timedelta(days=6, hours=23), now=now)
    out = _check(repo, holdout="B", pull_completed_at=now - timedelta(days=7), now=now)
    assert out["files_verified"] == 1
    with pytest.raises(LockError, match="pull_completed_at"):
        _check(repo, holdout="B")


def test_outcome_mapping():
    assert outcome_from_market("finalized", "no") == ("no", True, False)
    assert outcome_from_market("settled", "yes") == ("yes", True, False)
    assert outcome_from_market("finalized", "") == (None, False, True)
    assert outcome_from_market("finalized", None) == (None, False, True)
    assert outcome_from_market("active", "") == (None, False, False)
    assert outcome_from_market("closed", None) == (None, False, False)


def test_verdict_table():
    assert verdict(True, "pass") == "Supported (blind replication)"
    assert verdict(True, "unavailable") == "Supported (semi-blind only)"
    assert verdict(True, "underpowered") == "Supported (semi-blind only)"
    assert verdict(True, "fail") == "Not replicated out of sample"
    assert verdict(False, "pass") == "Not supported"
    assert verdict(None, None, validation_failed=True) == "Not supported (failed validation)"
