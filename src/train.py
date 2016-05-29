import nltk

import data

def computeLanguageBigramProbabilities():
   distributionDict = {}

   for lang in data.LanguageIds:
      langFile = data.getFilename("train", lang)

      with open(langFile, "r", encoding=data.Encoding) as trainFile:
         bigrams = nltk.bigrams(trainFile.read())

      dist = nltk.FreqDist(bigrams)
      goodTuring = nltk.SimpleGoodTuringProbDist(dist)
      distributionDict[lang] = goodTuring

   return distributionDict

def computeWordFrequencyDistributions():
   distributionDict = {}

   tokenizer = nltk.RegexpTokenizer(r'\w+')

   for lang in data.LanguageIds:
      langFile = data.getFilename("train", lang)

      with open(langFile, "r", encoding=data.Encoding) as trainFile:
         words = tokenizer.tokenize(trainFile.read())

      dist = nltk.FreqDist(words)
      goodTuring = nltk.SimpleGoodTuringProbDist(dist)
      distributionDict[lang] = goodTuring

   return distributionDict

