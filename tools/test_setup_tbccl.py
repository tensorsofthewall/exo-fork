# pyright: basic
"""tools/setup_tbccl.py: read-only checks, platform mapping, tag detection, missing-prerequisite reporting. No network, no sync. Run: python3 -m unittest tools/test_setup_tbccl.py"""

import contextlib
import io
import re
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import setup_tbccl as st


def run_main(argv):
    out = io.StringIO()
    with (
        mock.patch.object(sys, "argv", ["setup_tbccl.py", *argv]),
        contextlib.redirect_stdout(out),
        contextlib.redirect_stderr(out),
    ):
        code = st.main()
    return code, out.getvalue()


class SetupTbccl(unittest.TestCase):
    def test_platform_to_extra(self):
        with (
            mock.patch("platform.system", return_value="Linux"),
            mock.patch("platform.machine", return_value="x86_64"),
        ):
            self.assertEqual(st.mlx_extra()[0], "mlx-cuda13")
        with (
            mock.patch("platform.system", return_value="Darwin"),
            mock.patch("platform.machine", return_value="arm64"),
        ):
            self.assertEqual(st.mlx_extra()[0], "mlx")
        for system, machine in (
            ("Linux", "aarch64"),
            ("Darwin", "x86_64"),
            ("Windows", "AMD64"),
        ):
            with (
                mock.patch("platform.system", return_value=system),
                mock.patch("platform.machine", return_value=machine),
            ):
                self.assertIsNone(st.mlx_extra()[0])

    def test_unsupported_platform_is_explained_and_changes_nothing(self):
        with (
            mock.patch("platform.system", return_value="Linux"),
            mock.patch("platform.machine", return_value="aarch64"),
        ):
            code, text = run_main(["check"])
        self.assertEqual(code, 2)
        self.assertIn("no exo-tbccl wheel", text)
        self.assertIn("installs nothing here", text)

    def test_missing_tools_are_listed_with_what_to_do(self):
        with (
            mock.patch("platform.system", return_value="Linux"),
            mock.patch("platform.machine", return_value="x86_64"),
            mock.patch("shutil.which", return_value=None),
        ):
            code, text = run_main(["check"])
        self.assertEqual(code, 1)
        self.assertIn("MISSING  uv", text)
        self.assertIn("MISSING  Rust toolchain", text)
        self.assertIn("nothing was changed", text)

    def test_check_prints_the_locked_sync_and_never_installs(self):
        with (
            mock.patch("platform.system", return_value="Linux"),
            mock.patch("platform.machine", return_value="x86_64"),
            mock.patch("subprocess.run") as sub,
        ):
            sub.return_value = mock.Mock(returncode=0, stdout="x\n", stderr="")
            with mock.patch("shutil.which", return_value="/usr/bin/x"):
                code, text = run_main(["check"])
        self.assertEqual(code, 0)
        self.assertIn(
            "uv sync --no-dev --locked --extra mlx-cuda13 --extra tbccl", text
        )
        for call in sub.call_args_list:
            self.assertNotIn("sync", call.args[0], "check must not run a sync")

    def test_dashboard_api_only_hint(self):
        with (
            mock.patch("platform.system", return_value="Linux"),
            mock.patch("platform.machine", return_value="x86_64"),
            mock.patch("shutil.which", return_value=None),
        ):
            _, text = run_main(["check"])
        self.assertIn("EXO_DASHBOARD_DIR", text)

    def test_port_state_detects_a_listener(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("0.0.0.0", 0))
        s.listen(1)
        try:
            self.assertIn("in use", st.port_state("TCP", s.getsockname()[1]))
        finally:
            s.close()

    def test_doctor_without_an_environment(self):
        with mock.patch.object(st, "ROOT", Path(tempfile.mkdtemp())):
            code, text = run_main(["doctor"])
        self.assertEqual(code, 1)
        self.assertIn("no .venv", text)

    def test_script_has_no_network_or_system_mutation(self):
        source = Path(st.__file__).read_text()
        for forbidden in (
            r"\bifconfig\b",
            r"\bip (link|route|addr)\b",
            r"\biptables\b",
            r"\bpfctl\b",
            r"\bsudo\b",
            r"\bnetworksetup\b",
            r"\burllib\b",
            r"\brequests\b",
            r"brew install",
            r"\bapt(-get)?\b",
        ):
            self.assertIsNone(re.search(forbidden, source), forbidden)


if __name__ == "__main__":
    unittest.main()
