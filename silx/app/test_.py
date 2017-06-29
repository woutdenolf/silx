# coding: utf-8
# /*##########################################################################
# Copyright (C) 2016 European Synchrotron Radiation Facility
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
"""Launch unittests of the library"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "02/08/2017"

import sys
import os
import argparse
import logging
import unittest


class StreamHandlerUnittestReady(logging.StreamHandler):
    """The unittest class TestResult redefine sys.stdout/err to capture
    stdout/err from tests and to display them only when a test fail.

    This class allow to use unittest stdout-capture by using the last sys.stdout
    and not a cached one.
    """

    def emit(self, record):
        """
        :type record: logging.LogRecord
        """
        self.stream = sys.stderr
        super(StreamHandlerUnittestReady, self).emit(record)

    def flush(self):
        pass


# Use an handler compatible with unittests, else use_buffer is not working
for h in logging.root.handlers:
    logging.root.removeHandler(h)
logging.root.addHandler(StreamHandlerUnittestReady())

_logger = logging.getLogger(__name__)
"""Module logger"""


def main(argv):
    """
    Main function to launch the unittests as an application

    :param argv: Command line arguments
    :returns: exit status
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", default=0,
                        action="count", dest="verbose",
                        help="Increase verbosity. Option -v prints additional " +
                             "INFO messages. Use -vv for full verbosity, " +
                             "including debug messages and test help strings.")
    parser.add_argument("-x", "--no-gui", dest="gui", default=True,
                        action="store_false",
                        help="Disable the test of the graphical use interface")
    parser.add_argument("-g", "--no-opengl", dest="opengl", default=True,
                        action="store_false",
                        help="Disable tests using OpenGL")
    parser.add_argument("-o", "--no-opencl", dest="opencl", default=True,
                        action="store_false",
                        help="Disable the test of the OpenCL part")
    parser.add_argument("-l", "--low-mem", dest="low_mem", default=False,
                        action="store_true",
                        help="Disable test with large memory consumption (>100Mbyte")
    parser.add_argument("--qt-binding", dest="qt_binding", default=None,
                        help="Force using a Qt binding, from 'PyQt4', 'PyQt5', or 'PySide'")

    options = parser.parse_args(argv[1:])

    test_verbosity = 1
    use_buffer = True
    if options.verbose == 1:
        logging.root.setLevel(logging.INFO)
        _logger.info("Set log level: INFO")
        test_verbosity = 2
        use_buffer = False
    elif options.verbose > 1:
        logging.root.setLevel(logging.DEBUG)
        _logger.info("Set log level: DEBUG")
        test_verbosity = 2
        use_buffer = False

    if not options.gui:
        os.environ["WITH_QT_TEST"] = "False"

    if not options.opencl:
        os.environ["SILX_OPENCL"] = "False"

    if not options.opengl:
        os.environ["WITH_GL_TEST"] = "False"

    if options.low_mem:
        os.environ["SILX_TEST_LOW_MEM"] = "True"

    if options.qt_binding:
        binding = options.qt_binding.lower()
        if binding == "pyqt4":
            _logger.info("Force using PyQt4")
            import PyQt4.QtCore  # noqa
        elif binding == "pyqt5":
            _logger.info("Force using PyQt5")
            import PyQt5.QtCore  # noqa
        elif binding == "pyside":
            _logger.info("Force using PySide")
            import PySide.QtCore  # noqa
        else:
            raise ValueError("Qt binding '%s' is unknown" % options.qt_binding)

    # Run the tests
    runnerArgs = {}
    runnerArgs["verbosity"] = test_verbosity
    runnerArgs["buffer"] = use_buffer
    runner = unittest.TextTestRunner(**runnerArgs)

    import silx.test
    test_suite = unittest.TestSuite()
    test_suite.addTest(silx.test.suite())
    result = runner.run(test_suite)

    for test, reason in result.skipped:
        description = test.shortDescription() or ''
        _logger.warning('Skipped %s (%s): %s', test.id(), description, reason)

    if result.wasSuccessful():
        _logger.info("Test suite succeeded")
        exit_status = 0
    else:
        _logger.warning("Test suite failed")
        exit_status = 1
    return exit_status
