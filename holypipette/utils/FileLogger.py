# coding=utf-8
from __future__ import absolute_import

import collections

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

import functools
import logging
import datetime
import os
import traceback
import time
from types import MethodType

import param


from holypipette.interface.camera import CameraInterface
from holypipette.controller import TaskController
from holypipette.interface.patch import NumberWithUnit
from holypipette.interface.base import command

class FileLogger:
    def __init__(self, filename):
        self.filename = filename
        self.file = None

    def open(self):
        self.file = open(self.filename, 'w')

    def write_data(self, pressure, resistance, current, time_value):
        if self.file is None:
            self.open()
        self.file.write(f"{time_value}  {pressure}  {resistance}    {current}\n")

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None
    