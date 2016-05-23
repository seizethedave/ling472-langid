#!/usr/bin/env python3.4

import os

from data import DataDir, generateData

os.makedirs(DataDir, exist_ok=True)
print("Generating all data to %s..." % DataDir)
generateData()
print("Done.")
