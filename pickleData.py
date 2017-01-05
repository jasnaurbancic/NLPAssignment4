import pandas
import pickle

prefix = '../DOWNLOAD/twitter_download/'
files = ['test.txt', 'test2.txt']

dfs = (pandas.DataFrame.from_csv(prefix + f, sep='\t', index_col=False, header = None) for f in files)
data = pandas.concat(dfs, ignore_index=True)

# Go over all the data and remove Not Available tweets

data = data[data[2] != "Not Available"]

positive = len(data[data[1] == 'positive'])
negative = len(data[data[1] == 'negative'])
neutral = len(data[data[1] == 'neutral'])

fileObject = open('data.pickle', 'wb')  
pickle.dump(data, fileObject) 

print(''.join(['-']*80))
print("\t\tPositive Negative Neutral")
print("Absolute freq:")
print("\t\t", positive, "\t", negative, "\t", neutral)
print("Relative freq.:")
print("\t\t", positive/len(data), "\t", negative/len(data), "\t", neutral/len(data))
print(''.join(['-']*80))