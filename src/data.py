import re
import os
import random
from itertools import islice

from nltk import tokenize

Encoding = 'UTF-8'
EuroparlRoot = '/corpora/europarl/txt'

Languages = (
 # 2-tuples: (Europarl folder, nltk language)
 ('fr', 'french'),
 ('pt', 'portuguese'),
 ('it', 'italian'),
 ('es', 'spanish')
)

DataDir = os.path.join(
 os.path.realpath(os.path.dirname(__file__)), "data")

def getFilename(env, lang):
   return os.path.join(DataDir, "%s-%s.txt" % (lang, env))

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
   location = os.path.join(EuroparlRoot, europarlLanguage)

   for scriptFile in os.listdir(location):
      with open(os.path.join(location, scriptFile), encoding=Encoding) as f:
         normalizedText = normalizeEuroparlText(f.read())

      yield from tokenize.sent_tokenize(
       normalizedText, language=nltkLanguage)

def generateData():
   """
   Take the available language data and randomly stripe it into separate files
   for training, development, testing, etc.
   """
   # Set to None to use all fragments. This will allow us to tune in
   # an appropriate dataset size.
   fragmentLimit = 10000

   for europarlLang, nltkLang in Languages:
      trainFilename = getFilename('train', europarlLang)
      devFilename = getFilename('dev', europarlLang)
      testFilename = getFilename('test', europarlLang)

      with open(trainFilename, 'w', encoding=Encoding) as trainFile, \
       open(devFilename, 'w', encoding=Encoding) as devFile, \
       open(testFilename, 'w', encoding=Encoding) as testFile:

         files = (trainFile, devFile, testFile)

         fragments = islice(
          languageFragments(europarlLang, nltkLang), fragmentLimit)

         for fragment in fragments:
            print(fragment, file=random.choice(files))
