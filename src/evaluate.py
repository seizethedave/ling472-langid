#!/usr/bin/env python3.4

import os
import sys

from data import DataDir
from test import evaluate

if 1 == len(sys.argv) or sys.argv[1] not in {'test', 'dev'}:
   print("Execute with one of:")
   print("   ./evaluate.py test")
   print("   ./evaluate.py dev")
   sys.exit(1)

env = sys.argv[1]

evaluate(env)
