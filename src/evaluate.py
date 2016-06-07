#!/usr/bin/env python3.4

import os
import sys

from data import DataDir
from test import evaluate

if 1 == len(sys.argv):
   print("Execute with:")
   print("   ./evaluate.py foo [classifier]")
   print(
    "Where foo indicates a classify-foo.txt file in data/,")
   print(
    "and classifier is one of "
    "baseline|tandem|reluctant-tandem|bigram|frequency.)")
   sys.exit(1)

env = sys.argv[1]
classifier = sys.argv[2] if len(sys.argv) >= 3 else "tandem"
evaluate(env, classifier)
