"""PEP 517 backend wrapper around setuptools with EXDEV-safe artifact move."""

from __future__ import annotations

import errno
import os
import shutil
import sys
import tempfile
from typing import Iterable

from setuptools.build_meta import _BuildMetaBackend, _ConfigSettings, _file_with_extension, no_install_setup_requires


class CrossDeviceSafeBackend(_BuildMetaBackend):
    """Setuptools backend that tolerates cross-device rename errors in some CI/sandbox filesystems."""

    def _build_with_temp_dir(
        self,
        setup_command: Iterable[str],
        result_extension: str | tuple[str, ...],
        result_directory: str,
        config_settings: _ConfigSettings,
        arbitrary_args: Iterable[str] = (),
    ) -> str:
        result_directory = os.path.abspath(result_directory)
        os.makedirs(result_directory, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix=".tmp-", dir=result_directory) as tmp_dist_dir:
            sys.argv = [
                *sys.argv[:1],
                *self._global_args(config_settings),
                *setup_command,
                "--dist-dir",
                tmp_dist_dir,
                *arbitrary_args,
            ]
            with no_install_setup_requires():
                self.run_setup()

            result_basename = _file_with_extension(tmp_dist_dir, result_extension)
            result_path = os.path.join(result_directory, result_basename)
            src_path = os.path.join(tmp_dist_dir, result_basename)
            if os.path.exists(result_path):
                os.remove(result_path)
            try:
                os.rename(src_path, result_path)
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
                shutil.move(src_path, result_path)

        return result_basename


_backend = CrossDeviceSafeBackend()

get_requires_for_build_wheel = _backend.get_requires_for_build_wheel
get_requires_for_build_sdist = _backend.get_requires_for_build_sdist
prepare_metadata_for_build_wheel = _backend.prepare_metadata_for_build_wheel
build_wheel = _backend.build_wheel
build_sdist = _backend.build_sdist

if hasattr(_backend, "get_requires_for_build_editable"):
    get_requires_for_build_editable = _backend.get_requires_for_build_editable
if hasattr(_backend, "prepare_metadata_for_build_editable"):
    prepare_metadata_for_build_editable = _backend.prepare_metadata_for_build_editable
if hasattr(_backend, "build_editable"):
    build_editable = _backend.build_editable
