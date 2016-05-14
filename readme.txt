Install/setup:

Python version: Python 3.4 (the latest version that exists on Patas)

This means any Python scripts you run should be executed with "python3.4",
as just typing "python" will get you Python 2.x.

NLTK on Patas is already at v3.2.1 so let's just use that.

To download source for editing/running into a "ling472-langid"
folder beneath your home directory:

   cd ~
   git clone https://github.com/seizethedave/ling472-langid.git

NLTK's punkt data needs to be installed for the European language tokenization.
Evidently NLTK stores this data local to each user, so you must repeat
this process yourself:

   $ python3.4
   >>> import nltk
   >>> nltk.download()

This starts an interactive download tool.
Choose download ("d"), and when prompted, enter "punkt".
After install, exit from the NLTK downloader/Python.

===

Data generation:

There is a script that creates train/test/dev files for each of our supported languages.
This script is src/generate_data.py.
This script will create (if needed) a subfolder called /src/data and generate into it 12 files.
The data is sourced from the installed Europarl corpora, and includes ALL of it. This means
the Italian development set has 11+ million words. We can reduce this amount if desired.

To run the script:
   $ cd src
   $ ./generate_data.py
   
It will take ~10 minutes or so.

===

bookmarks/reading:

http://stackoverflow.com/questions/16379313/how-to-use-the-a-10-fold-cross-validation-with-naive-bayes-classifier-and-nltk
