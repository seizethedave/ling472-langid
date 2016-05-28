#!/usr/bin/env python3.4

import os

import data

os.makedirs(data.DataDir, exist_ok=True)
print("Generating training data to %s..." % data.DataDir)
data.generateTrainingData()
print("Done.")

print("Generating classificaion data to %s..." % data.DataDir)
data.generateClassificationData()
print("Done.")
