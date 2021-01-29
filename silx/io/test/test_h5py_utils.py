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
"""Tests for h5py utilities"""

__authors__ = ["W. de Nolf"]
__license__ = "MIT"
__date__ = "27/01/2020"


import unittest
import os
import shutil
import tempfile
import threading
import multiprocessing
from contextlib import contextmanager

from .. import h5py_utils


def _subprocess_context(queue, context, *args, **kw):
    try:
        with context(*args, **kw):
            queue.put(None)
            threading.Event().wait()
    except Exception:
        queue.put(None)
        raise


@contextmanager
def subprocess_context(context, *args, **kw):
    timeout = kw.pop("timeout", 10)
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_subprocess_context, args=(queue, context) + args, kwargs=kw
    )
    p.start()
    try:
        queue.get(timeout=timeout)
        yield
    finally:
        p.kill()
        p.join(timeout)


def subtests(test):
    def wrapper(self):
        for _ in self._subtests():
            with self.subTest(**self._subtest_options):
                test(self)

    return wrapper


class TestH5pyUtils(unittest.TestCase):
    H5PY1804_SOLVED = False

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _subtests(self):
        self._subtest_options = {"mode": "w"}
        self.filename_generator = self._filenames()
        yield
        if self.H5PY1804_SOLVED:
            # TODO: a concurrent reader with HDF5_USE_FILE_LOCKING=FALSE
            # no longer works with HDF5 >=1.10
            self._subtest_options = {"mode": "w", "libver": "latest"}
            self.filename_generator = self._filenames()
            yield

    def _filenames(self):
        i = 1
        while True:
            filename = os.path.join(self.test_dir, "file{}.h5".format(i))
            with self._open(filename) as f:
                self._write_hdf5_data(f)
            yield filename
            i += 1

    @contextmanager
    def _open(self, filename, **kwargs):
        kw = self._subtest_options
        kw.update(kwargs)
        with h5py_utils.File(filename, **kw) as f:
            yield f

    def _write_hdf5_data(self, f):
        f["check"] = True
        f.flush()

    def _assert_hdf5_data(self, f):
        self.assertTrue(f["check"][()])

    def _validate_hdf5_data(self, filename, swmr=False):
        with self._open(filename, mode="r") as f:
            self.assertEqual(f.swmr_mode, swmr)
            self._assert_hdf5_data(f)

    def new_filename(self):
        return next(self.filename_generator)

    @subtests
    def test_modes_single_process(self):
        orig = os.environ.get("HDF5_USE_FILE_LOCKING")
        filename1 = self.new_filename()
        self.assertEqual(orig, os.environ.get("HDF5_USE_FILE_LOCKING"))
        filename2 = self.new_filename()
        self.assertEqual(orig, os.environ.get("HDF5_USE_FILE_LOCKING"))
        with self._open(filename1, mode="r"):
            with self._open(filename2, mode="r"):
                pass
            for mode in ["w", "a"]:
                with self.assertRaises(RuntimeError):
                    with self._open(filename2, mode=mode):
                        pass
        self.assertEqual(orig, os.environ.get("HDF5_USE_FILE_LOCKING"))
        with self._open(filename1, mode="a"):
            for mode in ["w", "a"]:
                with self._open(filename2, mode=mode):
                    pass
            with self.assertRaises(RuntimeError):
                with self._open(filename2, mode="r"):
                    pass
        self.assertEqual(orig, os.environ.get("HDF5_USE_FILE_LOCKING"))

    @subtests
    def test_modes_multi_process(self):
        filename = self.new_filename()

        @contextmanager
        def submain(**kw):
            with self._open(filename, **kw) as f:
                if kw.get("mode") == "w":
                    # Re-create all datasets
                    self._write_hdf5_data(f)
                yield

        # File open by truncating writer
        with subprocess_context(submain, mode="w"):
            with self._open(filename, mode="r") as f:
                self._assert_hdf5_data(f)
            with self.assertRaises(OSError):
                with self._open(filename, mode="a") as f:
                    pass
            self._validate_hdf5_data(filename)
            if self.H5PY1804_SOLVED:
                with self.assertRaises(OSError):
                    with self._open(filename, mode="w") as f:
                        pass
                self._validate_hdf5_data(filename)

        # File open by appending writer
        with subprocess_context(submain, mode="a"):
            with self._open(filename, mode="r") as f:
                self._assert_hdf5_data(f)
            with self.assertRaises(OSError):
                with self._open(filename, mode="a") as f:
                    pass
            self._validate_hdf5_data(filename)
            if self.H5PY1804_SOLVED:
                with self.assertRaises(OSError):
                    with self._open(filename, mode="w") as f:
                        pass
                self._validate_hdf5_data(filename)

        # File open by reader
        with subprocess_context(submain, mode="r"):
            with self._open(filename, mode="r") as f:
                self._assert_hdf5_data(f)
            with self._open(filename, mode="a") as f:
                pass
            self._validate_hdf5_data(filename)
            if self.H5PY1804_SOLVED:
                with self._open(filename, mode="w") as f:
                    self._write_hdf5_data(f)
                self._validate_hdf5_data(filename)

        # File open by locking reader
        with subprocess_context(submain, mode="r", enable_file_locking=True):
            with self._open(filename, mode="r") as f:
                self._assert_hdf5_data(f)
            with self.assertRaises(OSError):
                with self._open(filename, mode="a") as f:
                    pass
            self._validate_hdf5_data(filename)
            if self.H5PY1804_SOLVED:
                with self.assertRaises(OSError):
                    with self._open(filename, mode="w") as f:
                        pass
                self._validate_hdf5_data(filename)

    @subtests
    @unittest.skipIf(
        not h5py_utils.HAS_SWMR, "SWMR not supported in this h5py library version"
    )
    def test_modes_multi_process_swmr(self):
        filename = self.new_filename()

        @contextmanager
        def submain(**kw):
            with self._open(filename, **kw) as f:
                yield

        with self._open(filename, mode="w", libver="latest") as f:
            self._write_hdf5_data(f)

        # File open by SWMR writer
        with subprocess_context(submain, mode="a", swmr=True):
            with self._open(filename, mode="r") as f:
                assert f.swmr_mode
                self._assert_hdf5_data(f)
            with self.assertRaises(OSError):
                with self._open(filename, mode="a") as f:
                    pass
            self._validate_hdf5_data(filename, swmr=True)
            if self.H5PY1804_SOLVED:
                with self.assertRaises(OSError):
                    with self._open(filename, mode="w") as f:
                        pass
                self._validate_hdf5_data(filename, swmr=True)

    @subtests
    def test_retry_defaults(self):
        filename = self.new_filename()
        with h5py_utils.open(filename) as fileobj:
            self.assertEqual(len(fileobj), 1)
        with self.assertRaises(h5py_utils.HDF5TimeoutError):
            with h5py_utils.open_item(filename, "/check", retry_timeout=0.2) as item:
                pass
        with h5py_utils.open_item(filename, "/check", validate=None) as item:
            self.assertTrue(item[()])
        with h5py_utils.open_top_level_items(filename, validate=None) as items:
            self.assertEqual(len(items), 1)
            self.assertTrue(items[0][()])


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestH5pyUtils))
    return test_suite


if __name__ == "__main__":
    unittest.main(defaultTest="suite")
