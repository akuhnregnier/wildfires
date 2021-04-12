# -*- coding: utf-8 -*-
import sys
import tempfile
from functools import partial
from pathlib import Path
from subprocess import DEVNULL, Popen

import pytest


@pytest.fixture
def tmp_dir():
    tmp_dir = tempfile.TemporaryDirectory()
    yield tmp_dir.name
    tmp_dir.cleanup()


def test_repeat_run_dependencies(tmp_dir):
    common_args = [
        sys.executable,
        str(Path(__file__).resolve().parent / "scripts" / "repeat_run_script.py"),
        "--tmp-dir",
        str(tmp_dir),
    ]
    popen_call = partial(
        Popen,
        stdin=DEVNULL,
        stdout=DEVNULL,
        stderr=DEVNULL,
    )
    # Upon the first call, the cache entries are created.
    proc = popen_call([*common_args, "--first-run"])
    proc.wait()
    assert not proc.returncode

    # The second call should then be able to access these.
    proc = popen_call(common_args)
    proc.wait()
    assert not proc.returncode
