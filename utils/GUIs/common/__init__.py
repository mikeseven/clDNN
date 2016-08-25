#!/usr/bin/python
from __future__ import print_function
import sys
import logging
from helper import *
from settings import *
from config_parser import ConfigParserWrapper



########################################################################################################################
class Flushfile(object):
    def __init__(self, fd):
        self.fd = fd

    ####################################################################################################################
    def write(self, x):
        ret = self.fd.write(x)
        self.fd.flush()
        return ret

    ####################################################################################################################
    def writelines(self, lines):
        ret = self.fd.writelines(lines)
        self.fd.flush()
        return ret

    ####################################################################################################################
    def flush(self):
        return self.fd.flush()

    ####################################################################################################################
    def close(self):
        return self.fd.close()

    ####################################################################################################################
    def fileno(self):
        return self.fd.fileno()

# replace the stdout
sys.stdout = Flushfile(sys.stdout)
logging.root.setLevel(logging.DEBUG)
logging.disable(logging.INFO)