import pandas
import pickle

prefix = '../DOWNLOAD/twitter_download/'
files = ['test.txt', 'test2.txt']

dfs = (pandas.DataFrame.from_csv(prefix + f, sep='\t', index_col=False) for f in files)
data = pandas.concat(dfs, ignore_index=True)

fileObject = open('data.pickle', 'wb')  
pickle.dump(data, fileObject) 