"""
Utils for testing and evaluating.
"""

from nltk.metrics.confusionmatrix import ConfusionMatrix

import train
import classify
from data import getClassificationFilename

def getClassifier(classifier):
   if 'base' == classifier:
      return classify.BaseClassifier()
   elif 'tandem' == classifier:
      print("Training frequencies...")
      freqDist = train.computeWordFrequencyDistributions()
      print("Done.")

      print("Training bigrams...")
      bigramDist = train.computeLanguageBigramProbabilities()
      print("Done.")
      return classify.TandemClassifier(freqDist, bigramDist)
   elif 'frequency' == classifier:
      print("Training frequencies...")
      freqDist = train.computeWordFrequencyDistributions()
      print("Done.")
      return classify.FrequencyClassifier(freqDist)
   elif 'bigram' == classifier:
      print("Training bigrams...")
      bigramDist = train.computeLanguageBigramProbabilities()
      print("Done.")
      return classify.BigramClassifier(bigramDist)

def evaluate(environment, classifierName):
   """
   Invoke classifier on all classification data for given environment,
   reporting precision/recall figures for each language encountered.
   """
   print("Evaluating classifier '%s' in environment '%s'." % (
    classifierName, environment))

   langMetrics = {
    'fr': [0, 0, 0],
    'es': [0, 0, 0],
    'it': [0, 0, 0],
    'pt': [0, 0, 0],
   }

   # Indices for true positive, false positive, false negative figures.
   (TP, FP, FN) = (0, 1, 2)

   # Track these for confusion matrix.
   gold = []
   results = []

   classifier = getClassifier(classifierName)

   with open(getClassificationFilename(environment), 'r') as f:
      for line in f:
         # Each line is <langid> <# sentences> <sentences>
         language, numSentences, fragment = line.split(" ", 2)

         classifiedLanguage = classifier.classify(fragment)

         # Track precision/recall stats.
         gold.append(language)
         results.append(classifiedLanguage)

         if language not in langMetrics:
            # For exotic langs not otherwise accounted for...
            langMetrics[language] = [0, 0, 0]

         if classifiedLanguage == language:
            langMetrics[language][TP] += 1
         else:
            langMetrics[classifiedLanguage][FP] += 1
            langMetrics[language][FN] += 1

   for lang, (TP, FP, FN) in sorted(langMetrics.items()):
      try:
         precision = TP / float(TP + FP)
         recall = TP / float(TP + FN)
         print("Language '%s': precision: %f, recall: %f" % (
          lang, precision, recall))
      except ZeroDivisionError:
         print("Language '%s' no precision/recall data available." % lang)

   print("")
   matrix = ConfusionMatrix(gold, results)
   print(matrix.pretty_format())
