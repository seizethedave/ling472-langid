"""
Utils for testing and evaluating.
"""

from nltk.metrics import scores

from classifier import classify
from data import getClassificationFilename

def evaluate(environment):
   controlResults = []
   classifierResults = []

   with open(getClassificationFilename(environment), 'r') as f:
      for line in f:
         # Each line is <langid> <# sentences> <sentences>
         language, numSentences, fragment = line.split(" ", 2)
         controlResults.append(language)
         classifierResults.append(classify(fragment))

      print("Accuracy is %.3f%%." % (
       scores.accuracy(controlResults, classifierResults) * 100
       ))
