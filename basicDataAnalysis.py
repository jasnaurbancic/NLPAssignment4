import pandas
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
from nltk import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import operator
from sklearn.metrics import confusion_matrix, f1_score

smileyReg = [("(:\)|:-\))+", "happy"),
          ("(:\(|:-\()+", "sad"),
          ("(;\)|;-\))+", "wink smirk"),
          ("(:'\)|:'-\))+", "happy cry"),
          ("(:'\(|:'-\()+", "sad cry"),
          ("(:D|:-D)+", "laugh"), 
          ("(:'D|:'-D)+", "laugh"),
          ("(:\*|:-\*)+", "kiss"),
          ("(:P|:-P)+", "playful"),
          ("(\>:‑\)|\>:\))+", "evil"),
          ("(:\/|:-\/)", "hesitant")]

def countSmiley(text):
  smils = 0
  #for s, m in smiley:
    #smils += text.count(s)
  return smils

def getResults(r):
  r['compound'] = -1
  m = max(r.items(), key=operator.itemgetter(1))[0]
  if (m == 'pos'):
    return 'positive'
  if (m == 'neg'):
    return 'negative'
  return 'neutral'

def preprocess(text):
  '''
    Lemmatize, remove stopwords, remove twitter handles and non-alphanumeric characters --- preprocess emojis!
    at the moment leaves stopwords in!!!
  '''
  #print(str(text))
  text = text.lower()
  #removes links, handles and hashtags
  text = re.sub("(&amp;)", " and ", text)
  text = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|#|(\w+:\/\/\S+)|rt"," ",text).split(" "))
  #for (s0, s1) in smileyReg:
    #text = re.sub(s0,s1,text)
    #print(text)
  #text = nltk.word_tokenize(text)
  #tokenizer = RegexpTokenizer(r'\w+')
  #text = tokenizer.tokenize(text)
  #stemmer = SnowballStemmer("english")
  #text = [stemmer.stem(w) for w in text if not w in stopwords.words('english')]
  #print(' '.join(text))
  return text
  

data = pickle.load(open( "train.pickle", "rb" ))
data[3] = data[2].map(lambda x: x.count("!"))
data[4] = data[2].map(lambda x: x.count("?"))
data[5] = data[2].map(lambda x: x.count("..."))
data[6] = data[2].map(lambda x: countSmiley(x))
data[7] = data[2].map(lambda x: preprocess(x))
'''
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
#posFreq.plot(50, cumulative=True)
positive = posFreq.most_common(100)

tmp = data[data[1] == "negative"][2].apply(preprocess).values.tolist()
for t in tmp:
  negative += t
negFreq = FreqDist(negative)
#negFreq.plot(50, cumulative=True)
negative = negFreq.most_common(100)

tmp = data[data[1] == "neutral"][2].apply(preprocess).values.tolist()
for t in tmp:
  neutral += t
neuFreq = FreqDist(neutral)
#neuFreq.plot(50, cumulative=True)
neutral = neuFreq.most_common(100)

for i in range(100):
  print(positive[i][0], positive[i][1]/posFreq.N(), negative[i][0], negative[i][1]/negFreq.N(), neutral[i][0], neutral[i][1]/neuFreq.N())
'''

#print(positive)
#for i in range(10):
  #preprocess(data[2].iloc[i])
  #print(data.iloc[i])
  
sid = SentimentIntensityAnalyzer()
results = data[7].apply(sid.polarity_scores)
results = results.apply(getResults)
#for i in range(10):
  #print(data[1].iloc[i], results.iloc[i])
  
cm = confusion_matrix(data[1].values, results.values, labels=["positive", "negative", "neutral"])
print(cm)
f1 = f1_score(data[1].values, results.values, labels = ["positive", "negative", "neutral"], average=None)
print("F1 score: ", f1)
