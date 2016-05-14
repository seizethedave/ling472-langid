import re
import os
from os import path
import random

from nltk import tokenize

Encoding = 'latin-1'
EuroparlRoot = '/corpora/europarl/txt'

Languages = (
 # 2-tuples: (Europarl folder, nltk language)
 ('fr', 'french'),
 ('pt', 'portuguese'),
 ('it', 'italian'),
 ('es', 'spanish')
)

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

   Let's try simply stripping anything inside angle brackets.

   Further potential normalization:
   - lower case
   """
   output = re.sub('<.+>', '', text)
   # Terminate unterminated sentences followed by "\n\n", like "comenzar"
   # above.
   output = re.sub(r'\w$\n\n', '. ', output, flags=re.MULTILINE)
   return output


def languageFragments(europarlLanguage, nltkLanguage):
   """
   Yields all fragments available for the specified language.
   Currently a fragment is a sentence, but this could be changed.
   """
   location = path.join(EuroparlRoot, europarlLanguage)

   for scriptFile in os.listdir(location):
      with open(path.join(location, scriptFile), encoding=Encoding) as f:
         normalizedText = normalizeEuroparlText(f.read())

      yield from tokenize.sent_tokenize(
       normalizedText, language=nltkLanguage)

def generateData(destinationDir):
   """
   Take the available language data and randomly stripe it into separate files
   for training, development, testing, etc.
   """
   for europarlLang, nltkLang in Languages:
      trainFilename = path.join(destinationDir, '%s-train.txt' % europarlLang)
      devFilename = path.join(destinationDir, '%s-dev.txt' % europarlLang)
      testFilename = path.join(destinationDir, '%s-test.txt' % europarlLang)

      with open(trainFilename, 'w+', encoding=Encoding) as trainFile, \
       open(devFilename, 'w+', encoding=Encoding) as devFile, \
       open(testFilename, 'w+', encoding=Encoding) as testFile:

         files = (trainFile, devFile, testFile)

         for fragment in languageFragments(europarlLang, nltkLang):
            print(fragment, file=random.choice(files))
