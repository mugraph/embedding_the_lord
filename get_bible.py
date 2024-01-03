
import urllib.request
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

print("Downloading ...")
url = "https://www.gutenberg.org/ebooks/10.txt.utf-8"
urllib.request.urlretrieve(url, filename="bible_raw.txt")

print("Reading ...")
with open("bible_raw.txt") as f:
    text = f.read()

print("Processing ...")
verses = text.split("\n\n")
verses = [sur.lower() for sur in verses if " " in sur]
verses = verses[10:-51]
verses = [sur.split(maxsplit=1)[1] for sur in verses]
text = " ".join(verses)
text = text.replace("\n", " ")
tokens = tokenizer.tokenize(text)
text = " ".join(tokens)

print("Saving ... ")
with open("bible_normalized.txt", "w") as f:
    f.write(text)

print("Done.")
