import pandas
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from nltk import FreqDist

smiley = [("(:\)|:-\))+", "happy"),
          ("(:\(|:-\()+", "sad"),
          ("(;\)|;-\))+", "wink smirk"),
          ("(:'\)|:'-\))+", "happy cry"),
          ("(:'\(|:'-\()+", "sad cry"),
          ("(:D|:-D)+", "laugh"), 
          ("(:'D|:'-D)+", "laugh"),
          ("(:\*|:-\*)+", "kiss"),
          ("(:P|:-P)+", "playful"),
          ("(\>:‑\)|\>:\))+", "evil")]

def preprocess(text):
  '''
    Lemmatize, remove stopwords, remove twitter handles and non-alphanumeric characters --- preprocess emojis!
    at the moment leaves stopwords in!!!
  '''
  #print(str(text))
  for (s0, s1) in smiley:
    text = re.sub(s0,s1,text)
    #print(text)
  text = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|#|(\w+:\/\/\S+)"," ",text).split(" "))
  text = nltk.word_tokenize(text)
  text = [w for w in text if not w in stopwords.words('english')]
  #print(' '.join(text))
  return text
  

data = pickle.load(open( "data.pickle", "rb" ))
positive = []
negative = []
neutral = []
#data = pandas.DataFrame(data)
#df = data[2].apply(preprocess)
#data.assign(lambda x: x[2].apply(preprocess))
tmp = data[data[1] == "positive"][2].apply(preprocess).values.tolist()
for t in tmp:
  positive += t
posFreq = FreqDist(positive)
posFreq.plot(50, cumulative=True)

tmp = data[data[1] == "negative"][2].apply(preprocess).values.tolist()
for t in tmp:
  negative += t
negFreq = FreqDist(negative)
negFreq.plot(50, cumulative=True)

tmp = data[data[1] == "neutral"][2].apply(preprocess).values.tolist()
for t in tmp:
  neutral += t
neuFreq = FreqDist(neutral)
neuFreq.plot(50, cumulative=True)
#print(positive)
#for i in range(10):
  #preprocess(data[2].iloc[i])
#print(data)

