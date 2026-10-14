# pyright: basic
"""The optional `tbccl` extra of pyproject.toml and its lock entries. No network. Run: python3 -m unittest tools/test_tbccl_extra.py"""

import re
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
LOCK = tomllib.loads((ROOT / "uv.lock").read_text())
VERSION = "0.3.0"


def packages(name):
    return [p for p in LOCK["package"] if p["name"] == name]


class TbcclExtra(unittest.TestCase):
    def test_extra_is_optional_and_pinned(self):
        project = PYPROJECT["project"]
        self.assertFalse(any("tbccl" in d for d in project["dependencies"]))
        extra = project["optional-dependencies"]["tbccl"]
        self.assertEqual(len(extra), 1)
        self.assertTrue(extra[0].startswith(f"exo-tbccl=={VERSION};"), extra)
        for other, reqs in project["optional-dependencies"].items():
            if other != "tbccl":
                self.assertFalse(any("tbccl" in r for r in reqs), other)

    def test_extra_is_limited_to_platforms_with_a_wheel(self):
        marker = PYPROJECT["project"]["optional-dependencies"]["tbccl"][0].split(
            ";", 1
        )[1]
        self.assertIn("platform_machine == 'x86_64'", marker)
        self.assertIn("platform_machine == 'arm64'", marker)
        self.assertNotIn("aarch64", marker)

    def test_sources_are_mutually_exclusive_release_wheels(self):
        sources = PYPROJECT["tool"]["uv"]["sources"]["exo-tbccl"]
        self.assertEqual(len(sources), 2)
        by_os = {
            ("linux" if "'linux'" in s["marker"] else "darwin"): s for s in sources
        }
        self.assertEqual(set(by_os), {"linux", "darwin"})
        self.assertIn("platform_machine == 'x86_64'", by_os["linux"]["marker"])
        self.assertIn("platform_machine == 'arm64'", by_os["darwin"]["marker"])
        base = f"https://github.com/tensorsofthewall/exo-tbccl/releases/download/v{VERSION}/exo_tbccl-{VERSION}-cp313-cp313-"
        self.assertTrue(by_os["linux"]["url"].startswith(base + "manylinux_"))
        self.assertTrue(
            by_os["linux"]["url"].endswith("x86_64.manylinux_2_28_x86_64.whl")
        )
        self.assertEqual(by_os["darwin"]["url"], base + "macosx_14_0_arm64.whl")
        self.assertEqual(PYPROJECT["project"]["requires-python"], "==3.13.*")

    def test_lock_has_both_wheels_with_hashes(self):
        locked = packages("exo-tbccl")
        self.assertEqual(len(locked), 2)
        for p in locked:
            self.assertEqual(p["version"], VERSION)
            self.assertEqual(len(p["wheels"]), 1)
            self.assertRegex(p["wheels"][0]["hash"], r"^sha256:[0-9a-f]{64}$")
            self.assertNotIn("dependencies", p)
        self.assertEqual(
            {p["wheels"][0]["url"] for p in locked},
            {s["url"] for s in PYPROJECT["tool"]["uv"]["sources"]["exo-tbccl"]},
        )

    def test_lock_records_the_extra(self):
        (exo,) = packages("exo")
        self.assertIn("tbccl", exo["metadata"]["provides-extras"])
        self.assertEqual(len(exo["optional-dependencies"]["tbccl"]), 2)

    def test_extra_resolves_to_one_wheel_or_nothing_per_platform(self):
        try:
            from packaging.markers import Marker
        except ImportError:
            self.skipTest("packaging is not installed")
        requirement = PYPROJECT["project"]["optional-dependencies"]["tbccl"][0]
        gate = Marker(requirement.split(";", 1)[1].strip())
        sources = PYPROJECT["tool"]["uv"]["sources"]["exo-tbccl"]
        expected = {
            ("linux", "x86_64"): "manylinux",
            ("linux", "aarch64"): None,
            ("darwin", "arm64"): "macosx_14_0_arm64",
            ("darwin", "x86_64"): None,
        }
        for (system, machine), wheel in expected.items():
            env = {"sys_platform": system, "platform_machine": machine}
            picked = [s["url"] for s in sources if Marker(s["marker"]).evaluate(env)]
            if wheel is None:
                self.assertFalse(gate.evaluate(env), (system, machine))
            else:
                self.assertTrue(gate.evaluate(env), (system, machine))
                self.assertEqual(len(picked), 1, (system, machine))
                self.assertIn(wheel, picked[0])

    def test_mlx_and_torch_pins_are_untouched(self):
        extras = PYPROJECT["project"]["optional-dependencies"]
        self.assertIn(
            "mlx-cuda-13==0.32.0; sys_platform == 'linux'", extras["mlx-cuda13"]
        )
        self.assertIn("torch==2.10.0; sys_platform == 'linux'", extras["mlx"])
        self.assertTrue(
            re.search(
                r"0e3ff3643b1c3719f78814b98e0d222afbad867c",
                str(PYPROJECT["tool"]["uv"]["sources"]["mlx"]),
            )
        )


if __name__ == "__main__":
    unittest.main()
