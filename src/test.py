"""
Utils for testing and evaluating.
"""

import train
from classifier import BigramClassifier, TandemClassifier
from data import getClassificationFilename

def evaluate(environment):
   """
   Invoke classifier on all classification data for given environment,
   reporting precision/recall figures for each language encountered.
   """
   langMetrics = {
    'fr': [0, 0, 0],
    'es': [0, 0, 0],
    'it': [0, 0, 0],
    'pt': [0, 0, 0],
   }

   print("Training frequencies...")
   freqDist = train.computeWordFrequencyDistributions()
   print("Done.")

   print("Training bigrams...")
   bigramDist = train.computeLanguageBigramProbabilities()
   print("Done.")

   c = TandemClassifier(freqDist, bigramDist)
   #bigramClassifier = BigramClassifier(distributions)
   # Indices for true positive, false positive, false negative figures.
   (TP, FP, FN) = (0, 1, 2)

   limit = 200

   with open(getClassificationFilename(environment), 'r') as f:
      for line in f:
         if limit > 0:
            limit -= 1
         else:
            break
         # Each line is <langid> <# sentences> <sentences>
         language, numSentences, fragment = line.split(" ", 2)

         classifiedLanguage = c.classify(fragment)

         print("Classified\n\t\"%s\"\n\t as %s." % (
          fragment, classifiedLanguage))

         # Track precision/recall stats.

         if classifiedLanguage == language:
            langMetrics[language][TP] += 1
         else:
            langMetrics[classifiedLanguage][FP] += 1
            langMetrics[language][FN] += 1

   for lang, (TP, FP, FN) in sorted(langMetrics.items()):
      if TP + FP == 0 or TP + FN == 0:
         print("Language '%s' no precision/recall data available." % lang)
         continue

      precision = TP / float(TP + FP)
      recall = TP / float(TP + FN)
      print("Language '%s': precision: %f, recall: %f" % (
       lang, precision, recall))
