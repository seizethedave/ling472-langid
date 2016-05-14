Python version:

Python 3.4 (the latest version that exists on Patas)

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

bookmarks/reading:

http://stackoverflow.com/questions/16379313/how-to-use-the-a-10-fold-cross-validation-with-naive-bayes-classifier-and-nltk
