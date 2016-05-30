import re
import os
import random
from itertools import islice

from nltk import tokenize

Encoding = 'UTF-8'
EuroparlRoot = '/corpora/europarl/txt'

TrainingSentenceLimit = 1000
ClassificationFragmentLimit = 1000

Languages = (
 # 2-tuples: (language ID, language name)
 # Conveniently, Europarl stores data by the first one; NLTK identifies
 # language by the second.
 ('fr', 'french'),
 ('pt', 'portuguese'),
 ('it', 'italian'),
 ('es', 'spanish')
)

LanguageIds = tuple(langId for langId, langName in Languages)

DataDir = os.path.join(
 os.path.realpath(os.path.dirname(__file__)), "data")

def getFilename(env, lang):
   return os.path.join(DataDir, "%s-%s.txt" % (lang, env))

def getClassificationFilename(env):
   return os.path.join(DataDir, "classify-%s.txt" % env)

def normalizeEuroparlText(text):
   """
   The Europarl texts are formatted in a crude XMLish structure, but not valid
   XML with closing tags. It looks like this:

      <SPEAKER ID=3 NAME="El Presidente"> Tomamos nota, Sr. Balfe.  <P> El Acta
      queda aprobada

      <SPEAKER ID=4 LANGUAGE="EN" NAME="Hardstaff"> Señor Presidente, deseo
      llamar su atención sobre el hecho de que los debates sobre agricultura
      comenzar

      <SPEAKER ID=5 NAME="El Presidente"> Señora Hardstaff, los servicios de
      traducción me informan de que realmente ha habido un problema técni

   """
   # Eliminate anything inside angle brackets.
   output = re.sub('<.+>', '', text)
   # Terminate unterminated sentences followed by "\n\n", like "comenzar"
   # above.
   output = re.sub(r'\w$\n\n', '. ', output, flags=re.MULTILINE)
   # Finally, strip newlines.
   output = re.sub(r'\n+', ' ', output, flags=re.MULTILINE)
   # And lowercase everything.
   return output.lower()

def languageFragments(europarlLanguage, nltkLanguage):
   """
   Yields all fragments available for the specified language.
   Currently a fragment is a sentence, but this could be changed.
   """
   langFolder = os.path.join(EuroparlRoot, europarlLanguage)

   for scriptFile in os.listdir(langFolder):
      with open(os.path.join(langFolder, scriptFile), encoding=Encoding) as f:
         normalizedText = normalizeEuroparlText(f.read())

      yield from tokenize.sent_tokenize(
       normalizedText, language=nltkLanguage)

def generateAllData():
   os.makedirs(DataDir, exist_ok=True)

   languageSentenceIterators = {
    langId: iter(languageFragments(langId, langName))
    for langId, langName in Languages
   }

   print("Generating training data to %s..." % DataDir)

   generateTrainingData({
    langId: islice(it, TrainingSentenceLimit)
    for langId, it in languageSentenceIterators.items()
   })

   print("Done.")

   print("Generating classification data to %s..." % DataDir)

   # Give the rest of the data to the classification data generator.
   generateClassificationData(languageSentenceIterators)

   print("Done.")

def generateTrainingData(languageSentenceIterators):
   for langId, iterator in languageSentenceIterators.items():
      trainFilename = getFilename('train', langId)

      with open(trainFilename, 'w', encoding=Encoding) as trainFile:
         for sentence in iterator:
            print(sentence, file=trainFile)

def generateClassificationData(languageSentenceIterators):
   """
   We generate all environments' classification sample data here.
   Each line in the resulting file will be formatted as
   <language ID> <# sentences> <sentences>

   Sample:
      pt 1 senhora presidente, lamento ter de repetir sempre a mesma coisa em r
      es 3 mi información sólo concierne al sr. holmes. orden de los trabajo. d
      es 1 señora presidenta, lamento tener que decir una y otra vez lo mismo a
      it 6 ormai l'eccezione comincia a diventare la regola, visto che, essendo
      pt 7 como sabe, fazemos tudo o que podemos para melhorar as coisas. creio
      es 3 cada vez que protesto se me responde que es una excepción. pero, ent

   Each line represents one sample input. We generate a variety of
   input lengths.
   """
   sentenceLimits = (1, 20)

   devFilename = getClassificationFilename("dev")
   testFilename = getClassificationFilename("test")

   with open(devFilename, "w", encoding=Encoding) as devFile, \
         open(testFilename, "w", encoding=Encoding) as testFile:
      outputFiles = (devFile, testFile)

      for i in range(ClassificationFragmentLimit):
         fragmentSize = random.randint(*sentenceLimits)

         # Randomly choose a language, pull a fragment of the desired size.

         languageId = random.choice(LanguageIds)
         iterator = languageSentenceIterators[languageId]
         fragments = [next(iterator) for n in range(fragmentSize)]

         print(u"%s %d %s" %
          (languageId, fragmentSize, u" ".join(fragments)),
          file=random.choice(outputFiles))
