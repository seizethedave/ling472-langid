#!/usr/bin/env python3.4

import os
import sys

from data import DataDir
from test import evaluate

if 1 == len(sys.argv):
   print("Execute with:")
   print("   ./evaluate.py foo")
   print(
    "   (where foo indicates a file in data/ with name classify-foo.txt.)")
   sys.exit(1)

env = sys.argv[1]
evaluate(env)
