# pyright: basic
"""Tests of check_upstream_drift.py on throwaway git repositories (no network). Run: python3 -m unittest tools/test_check_upstream_drift.py"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_upstream_drift as drift

ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.org",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.org",
}


def git(cwd, *args):
    p = subprocess.run(
        ["git", "-C", cwd, *args], capture_output=True, text=True, env=ENV, check=True
    )
    return p.stdout.strip()


def commit(cwd, name, text):
    Path(cwd, name).parent.mkdir(parents=True, exist_ok=True)
    Path(cwd, name).write_text(text)
    git(cwd, "add", "-A")
    git(cwd, "commit", "-qm", f"change {name}")
    return git(cwd, "rev-parse", "HEAD")


class Drift(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = self.tmp.name
        git(self.repo, "init", "-q", "-b", "main")
        self.base = commit(self.repo, "a.py", "1")
        commit(self.repo, "seam.py", "1")
        git(self.repo, "tag", "integration")
        self.addCleanup(self.tmp.cleanup)
        cwd = os.getcwd()
        os.chdir(self.repo)
        self.addCleanup(os.chdir, cwd)

    def test_no_drift(self):
        git(self.repo, "update-ref", "refs/remotes/up/main", self.base)
        r = drift.analyse("main", "refs/remotes/up/main", self.base, "integration")
        self.assertEqual((r["ahead"], r["behind"], r["overlap"]), (1, 0, []))
        self.assertEqual(r["touch_points"], ["seam.py"])

    def test_upstream_moved_without_touching_the_integration(self):
        git(self.repo, "checkout", "-q", "-b", "up", self.base)
        up = commit(self.repo, "other.py", "2")
        git(self.repo, "checkout", "-q", "main")
        git(self.repo, "update-ref", "refs/remotes/up/main", up)
        r = drift.analyse("main", "refs/remotes/up/main", self.base, "integration")
        self.assertEqual(
            (r["behind"], r["upstream_files"], r["overlap"]), (1, ["other.py"], [])
        )

    def test_upstream_change_overlapping_the_integration(self):
        git(self.repo, "checkout", "-q", "-b", "up", self.base)
        commit(self.repo, "seam.py", "upstream edit")
        up = commit(self.repo, "other.py", "2")
        git(self.repo, "checkout", "-q", "main")
        git(self.repo, "update-ref", "refs/remotes/up/main", up)
        r = drift.analyse("main", "refs/remotes/up/main", self.base, "integration")
        self.assertEqual(r["overlap"], ["seam.py"])
        self.assertIn(
            "seam.py",
            drift.render(
                r,
                type(
                    "A",
                    (),
                    {
                        "name": "x",
                        "upstream_branch": "main",
                        "base": self.base,
                        "tag": "integration",
                    },
                )(),
            ),
        )


if __name__ == "__main__":
    unittest.main()
