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