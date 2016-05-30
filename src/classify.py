import random
import operator

import nltk

def baseClassifier(text):
   return 'fr'

def weightedRandomClassifier(text):
   # Impose some random weighting.
   return random.choice(['fr'] * 7 + ['it'] * 4 + ['es'] * 3 + ['pt'] * 2)

class TandemClassifier(object):
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

      """
      print("Freq scores: %r" % (freqScores))
      print("Bigram scores: %r" % (bigramScores))
      print("Overall scores: %r" % (results))
      """

      return results
      

class FrequencyClassifier(object):
   def __init__(self, distributions):
      self.distributions = distributions

   def classify(self, text):
      tokenizer = nltk.RegexpTokenizer(r'\w+')
      words = tokenizer.tokenize(text)

      bestLang = ''
      bestScore = -713047205702707

      for lang, distribution in self.distributions.items():
         hapaxProb = distribution.logprob(
          distribution.freqdist().hapaxes()[0])
         langScore = 0

         for word in words:
            if word in distribution.freqdist():
               langScore += distribution.logprob(word)
            else:
               langScore += hapaxProb

         if langScore > bestScore:
            bestScore = langScore
            bestLang = lang

      return bestLang

   def classify2(self, text):
      tokenizer = nltk.RegexpTokenizer(r'\w+')
      words = tokenizer.tokenize(text)

      results = { }

      for lang, distribution in self.distributions.items():
         hapaxProb = distribution.logprob(
          distribution.freqdist().hapaxes()[0])
         langScore = 0

         for word in words:
            if word in distribution.freqdist():
               langScore += distribution.logprob(word)
            else:
               langScore += hapaxProb

         results[lang] = langScore

      return results

class BigramClassifier(object):
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
