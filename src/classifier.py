import random

def baseClassifier(text):
   return 'fr'

def weightedRandomClassifier(text):
   # Impose some random weighting.
   return random.choice(['fr'] * 6 + ['it'] * 4 + ['es'] * 3 + ['pt'] * 2)

classify = weightedRandomClassifier
