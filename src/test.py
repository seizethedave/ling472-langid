"""
Utils for testing and evaluating.
"""

from nltk.metrics import scores

from classifier import classify
from data import getFilename

def test():
   assert 'french' == classify('Au revoire')

def evaluate(environment):
   languages = ('fr', 'it', 'es', 'pt')

   # Tally per-language results, and add them to overall results.

   overallControlResults = []
   overallClassifierResults = []

   for language in languages:
      controlResults = []
      classifierResults = []

      with open(getFilename(environment, language), 'r') as f:
         for fragment in f:
            controlResults.append(language)
            classifierResults.append(classify(fragment))

         print("For language %s, accuracy is %f." % (
          language,
          scores.accuracy(controlResults, classifierResults)
          ))

      overallControlResults.extend(controlResults)
      overallClassifierResults.extend(classifierResults)


   print("For overall environment %s, accuracy is %f." % (
    environment,
    scores.accuracy(overallControlResults, overallClassifierResults)
    ))

if '__main__' == __name__:
   test()
