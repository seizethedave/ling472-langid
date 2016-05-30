#!/usr/bin/env python3.4

"""
Converts one of our classify data files to one composed of N words rather than
N sentences, so that we can evaluate based on some VERY short inputs.

Usage: make_word_file.py < classify-dev.txt > classify-short.txt
"""

import sys

for line in sys.stdin:
   lang, size, text = line.split(u" ", 2)
   size = int(size)
   shortText = u" ".join(text.split(u" ")[:size])
   print(u"%s %d %s" % (lang, size, shortText))
