# coding: utf-8
# /*##########################################################################
# Copyright (C) 2016-2017 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ############################################################################*/
"""
This module contains wrapper for h5py. The exposed layout is
as close as possible to the original file format.
"""

__authors__ = ["W. de Nolf"]
__license__ = "MIT"
__date__ = "27/01/2020"


import os
import time
from functools import wraps
from contextlib import contextmanager
import errno
import h5py

from .._version import calc_hexversion

H5PY_HEX_VERSION = calc_hexversion(*h5py.version.version_tuple[:3])
HDF5_HEX_VERSION = calc_hexversion(*h5py.version.hdf5_version_tuple[:3])

HAS_SWMR = HDF5_HEX_VERSION >= calc_hexversion(*h5py.get_config().swmr_min_hdf5_version)
HAS_TRACK_ORDER = H5PY_HEX_VERSION >= calc_hexversion(2, 9, 0)

RETRY_PERIOD = 0.01


class HDF5TimeoutError(TimeoutError):
    pass


class HDF5RetryError(RuntimeError):
    pass


def isErrno(e, errno):
    """
    :param OSError e:
    :returns bool:
    """
    return "errno = {}".format(errno) in str(e)


def retry(timeout=None, context=False, retry_period=None, **kw):
    """Decorate method that open+read an HDF5 file that is being written too.
    When HDF5 IO fails (because the writer is modifying the file) the method
    or context manager will be retried.

    The wrapped method or context manager needs to be idempotent.

    The wrapped method can be calld with `method(..., retry_timeout=1)`
    which will overwrite the decorator's `timeout` argument.

    :param num timeout:
    :param bool context: the wrapped method is a context manager
    :param num retry_period: sleep before retyr
    :param **kw: named arguments for `File`
    """

    if retry_period is None:
        retry_period = RETRY_PERIOD

    def decorator(method):
        @wraps(method)
        def wrapper(filename, *args, **_kw):
            _timeout = _kw.pop("retry_timeout", timeout)
            has_timeout = _timeout is not None
            exception = None

            if has_timeout:
                t0 = time.time()
            while True:
                try:
                    with File(filename, **kw) as f:
                        result = method(f, *args, **_kw)
                        if context:
                            yield next(result)
                            return
                        else:
                            return result
                except HDF5RetryError as e:
                    exception = e
                except HDF5TimeoutError:
                    raise
                except OSError as e:
                    # TODO: needs to come from h5py to pass
                    exception = e
                except RuntimeError as e:
                    # TODO: needs to come from h5py to pass
                    exception = e
                except KeyError as e:
                    if "doesn't exist" in str(e):
                        raise
                    exception = e
                    # TODO: needs to come from h5py to pass
                if retry_period >= 0:
                    time.sleep(retry_period)
                if has_timeout and (time.time() - t0) > _timeout:
                    raise HDF5TimeoutError from exception

        if context:
            return contextmanager(wrapper)
        else:
            return wrapper

    return decorator


def default_validate(h5item):
    try:
        return "end_time" in h5item
    except Exception:
        return False


@retry(timeout=None, context=True)
def open_top_level_items(h5file, validate=default_validate):
    """Yields all valid top-level HDF5 items (retry until all top-level
    items can be retrieved).

    :param str h5file:
    :param callable or None validate:
    :yields list:
    """
    h5items = [h5file[name] for name in h5file["/"]]
    if callable(validate):
        h5items = [h5item for h5item in h5items if validate(h5item)]
    yield h5items


@retry(timeout=None, context=True)
def open_item(h5file, name, validate=default_validate):
    """Yields an HDF5 item (retry until it is valid).

    :param str h5file:
    :param str name:
    :param callable or None validate:
    :yields Group or Dataset:
    """
    h5item = h5file[name]
    if callable(validate):
        if not validate(h5item):
            raise HDF5RetryError
    yield h5item


@retry(timeout=None, context=True)
def open(h5file):
    """Yield and HDF5 file object (retry until we can open the file).

    :param str h5file:
    :yields File:
    """
    yield h5file


class File(h5py.File):
    _HDF5_FILE_LOCKING = None
    _NOPEN = 0
    _SWMR_LIBVER = "latest"

    def __init__(
        self,
        filename,
        mode=None,
        enable_file_locking=None,
        swmr=None,
        libver=None,
        **kwargs
    ):
        """
        :param str filename:
        :param str or None mode: read-only by default
        :param bool or None enable_file_locking: by default it is disabled for `mode='r'`
                                                 and `swmr=False` and enabled for all
                                                 other modes.
        :param bool or None swmr: try both modes when `mode='r'` and `swmr=None`
        :param **kwargs: see `h5py.File.__init__`
        """
        if mode is None:
            mode = "r"
        elif mode not in ("r", "w", "w-", "x", "a"):
            raise ValueError("invalid mode {}".format(mode))
        if not HAS_SWMR:
            swmr = False

        if enable_file_locking is None:
            enable_file_locking = bool(mode != "r" or swmr)
        if self._NOPEN:
            self._check_locking_env(enable_file_locking)
        else:
            self._set_locking_env(enable_file_locking)

        if swmr and libver is None:
            libver = self._SWMR_LIBVER

        if HAS_TRACK_ORDER:
            kwargs.setdefault("track_order", True)
        try:
            super().__init__(filename, mode=mode, swmr=swmr, libver=libver, **kwargs)
        except OSError as e:
            #   wlock   wSWMR   rlock   rSWMR   OSError: Unable to open file (...)
            # 1 TRUE    FALSE   FALSE   FALSE   -
            # 2 TRUE    FALSE   FALSE   TRUE    -
            # 3 TRUE    FALSE   TRUE    FALSE   unable to lock file, errno = 11, error message = 'Resource temporarily unavailable'
            # 4 TRUE    FALSE   TRUE    TRUE    unable to lock file, errno = 11, error message = 'Resource temporarily unavailable'
            # 5 TRUE    TRUE    FALSE   FALSE   file is already open for write (may use <h5clear file> to clear file consistency flags)
            # 6 TRUE    TRUE    FALSE   TRUE    -
            # 7 TRUE    TRUE    TRUE    FALSE   file is already open for write (may use <h5clear file> to clear file consistency flags)
            # 8 TRUE    TRUE    TRUE    TRUE    -
            if (
                mode == "r"
                and swmr is None
                and "file is already open for write" in str(e)
            ):
                # Try reading in SWMR mode (situation 5 and 7)
                swmr = True
                if libver is None:
                    libver = self._SWMR_LIBVER
                super().__init__(
                    filename, mode=mode, swmr=swmr, libver=libver, **kwargs
                )
            else:
                raise
        else:
            self._add_nopen(1)
            try:
                if mode != "r" and swmr:
                    # Try setting writer in SWMR mode
                    self.swmr_mode = True
            except Exception:
                self.close()
                raise

    @classmethod
    def _add_nopen(cls, v):
        cls._NOPEN = max(cls._NOPEN + v, 0)

    def close(self):
        super().close()
        self._add_nopen(-1)
        if not self._NOPEN:
            self._restore_locking_env()

    def _set_locking_env(self, enable):
        self._backup_locking_env()
        if enable:
            os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"
        elif enable is None:
            try:
                del os.environ["HDF5_USE_FILE_LOCKING"]
            except KeyError:
                pass
        else:
            os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    def _get_locking_env(self):
        v = os.environ.get("HDF5_USE_FILE_LOCKING")
        if v == "TRUE":
            return True
        elif v is None:
            return None
        else:
            return False

    def _check_locking_env(self, enable):
        if enable != self._get_locking_env():
            if enable:
                raise RuntimeError(
                    "Close all HDF5 files before enabling HDF5 file locking"
                )
            else:
                raise RuntimeError(
                    "Close all HDF5 files before disabling HDF5 file locking"
                )

    def _backup_locking_env(self):
        v = os.environ.get("HDF5_USE_FILE_LOCKING")
        if v is None:
            self._HDF5_FILE_LOCKING = None
        else:
            self._HDF5_FILE_LOCKING = v == "TRUE"

    def _restore_locking_env(self):
        self._set_locking_env(self._HDF5_FILE_LOCKING)
        self._HDF5_FILE_LOCKING = None
