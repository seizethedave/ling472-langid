#!/usr/bin/env python

import sys
import os

from data import generateData

thisDir = os.path.realpath(os.path.dirname(__file__))
dataDir = os.path.join(thisDir, "data")
os.makedirs(dataDir, exist_ok=True)
print("Generating all data to %s..." % dataDir)
generateData(dataDir)
print("Done.")
