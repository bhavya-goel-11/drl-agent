"""
Git Sync Utilities — Commit & push artifacts from Colab back to GitHub.

All functions are designed to be safe: if a push fails (e.g. no PAT configured
locally), they log a warning and return False rather than crashing the pipeline.
"""

import subprocess
import os
from pathlib import Path
from loguru import logger

# Project root is two levels up from this file (utils/git_sync.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def setup_git_identity(name: str = "Colab Pipeline", email: str = "colab@pipeline.bot"):
    """Configure git user identity for commits made from Colab."""
    _run_git(["config", "user.name", name])
    _run_git(["config", "user.email", email])
    logger.info(f"Git identity set: {name} <{email}>")


def git_sync(paths: list, message: str, repo_dir: str = None) -> bool:
    """Stage specific paths, commit, and push to origin main.

    Parameters
    ----------
    paths : list of str
        Relative paths (from repo root) to stage. E.g. ["data/", "models/"]
    message : str
        Commit message.
    repo_dir : str, optional
        Override the repo working directory. Defaults to PROJECT_ROOT.

    Returns
    -------
    bool
        True if the push succeeded, False otherwise.
    """
    cwd = repo_dir or str(PROJECT_ROOT)

    try:
        # Stage the specified paths
        for p in paths:
            result = _run_git(["add", p], cwd=cwd)
            if result.returncode != 0:
                logger.warning(f"git add failed for {p}: {result.stderr}")

        # Check if there are staged changes
        status = _run_git(["diff", "--cached", "--quiet"], cwd=cwd)
        if status.returncode == 0:
            logger.info("No changes to commit — skipping push.")
            return True

        # Commit
        commit_result = _run_git(["commit", "-m", message], cwd=cwd)
        if commit_result.returncode != 0:
            logger.warning(f"git commit failed: {commit_result.stderr}")
            return False

        # Push
        push_result = _run_git(["push", "origin", "main"], cwd=cwd)
        if push_result.returncode != 0:
            logger.warning(f"git push failed: {push_result.stderr}")
            return False

        logger.success(f"Git sync complete: {message}")
        return True

    except Exception as e:
        logger.warning(f"Git sync failed (non-fatal): {e}")
        return False


# ── Convenience Wrappers ──────────────────────────────────────────────────────

def sync_data(repo_dir: str = None) -> bool:
    """Commit and push the data/ directory."""
    return git_sync(
        paths=["data/"],
        message="[pipeline] Update processed_data.parquet",
        repo_dir=repo_dir,
    )


def sync_model(episode: int = 0, repo_dir: str = None) -> bool:
    """Commit and push the models/ directory."""
    return git_sync(
        paths=["models/"],
        message=f"[pipeline] Checkpoint model weights (episode {episode})",
        repo_dir=repo_dir,
    )


def sync_reports(repo_dir: str = None) -> bool:
    """Commit and push the reports/ directory."""
    return git_sync(
        paths=["reports/"],
        message="[pipeline] Update backtest reports",
        repo_dir=repo_dir,
    )


# ── Internal Helper ───────────────────────────────────────────────────────────

def _run_git(args: list, cwd: str = None) -> subprocess.CompletedProcess:
    """Run a git command and return the CompletedProcess."""
    cwd = cwd or str(PROJECT_ROOT)
    return subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
    )
