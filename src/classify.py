import operator
from statistics import stdev

import nltk

Unknown = "*UNK*"
ConfidenceBoundary = 100.0

class BaseClassifier:
   def classify(self, text):
      return 'fr'

class TandemClassifier:
   def __init__(self, freqDistributions, bigramDistributions):
      self.freqClassifier = FrequencyClassifier(freqDistributions)
      self.bigramClassifier = BigramClassifier(bigramDistributions)

   def classify(self, text):
      return max(self.classify2(text).items(), key=operator.itemgetter(1))[0]

   def classify2(self, text):
      freqScores = self.freqClassifier.classify2(text)
      bigramScores = self.bigramClassifier.classify2(text)

      results = { }

      for lang in freqScores.keys():
         results[lang] = freqScores[lang] + bigramScores[lang]

      return results

class ReluctantTandemClassifier(TandemClassifier):
   """
   Specialized TandemClassifier that will produce `Unknown` when not confident
   about the result.
   """
   def classify(self, text):
      probs = super().classify2(text)
      probNumbers = probs.values()
      
      confidence = (max(probNumbers) - min(probNumbers)) * stdev(probNumbers)

      if confidence < ConfidenceBoundary:
         return Unknown

      return max(probs.items(), key=operator.itemgetter(1))[0]

class FrequencyClassifier:
   def __init__(self, distributions):
      self.distributions = distributions
      self.tokenizer = nltk.RegexpTokenizer(r'\w+')

   def classify(self, text):
      words = self.tokenizer.tokenize(text)
      bestLang = ''
      bestScore = -713047205702707

      for lang, distribution in self.distributions.items():
         langScore = sum(distribution.logprob(word) for word in words)

         if langScore > bestScore:
            bestScore = langScore
            bestLang = lang

      return bestLang

   def classify2(self, text):
      words = self.tokenizer.tokenize(text)

      results = { }

      for lang, distribution in self.distributions.items():
         langScore = sum(distribution.logprob(word) for word in words)
         results[lang] = langScore

      return results

class BigramClassifier:
   def __init__(self, distributions):
      self.distributions = distributions

   def classify(self, text):
      """
      Convert to bigrams, calculate probability.
      """
      grams = list(nltk.bigrams(text))

      def scoreLanguageTextPair(pair):
         language, distribution = pair
         return sum(distribution.logprob(bigram) for bigram in grams)

      return max(self.distributions.items(), key=scoreLanguageTextPair)[0]

   def classify2(self, text):
      grams = list(nltk.bigrams(text))

      def scoreLanguageTextPair(pair):
         language, distribution = pair
         return sum(distribution.logprob(bigram) for bigram in grams)

      values = { }

      for language, distribution in self.distributions.items():
         values[language] = scoreLanguageTextPair((language, distribution))

      return values
