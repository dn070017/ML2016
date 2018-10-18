
# coding: utf-8

# In[254]:

def read_doc(path):
    doc_file = open(path, 'r')
    doc_content = doc_file.read()
    doc_content = doc_content.lower()
    translator = str.maketrans({key: ' ' for key in string.punctuation})
    doc_no_punctuation = doc_content.translate(translator)
    translator = str.maketrans('', '', string.digits)
    doc_remove_digit = doc_no_punctuation.translate(translator)
    doc_tokens = nltk.word_tokenize(doc_remove_digit)
    doc_filtered = [w for w in doc_tokens if not w in stopwords.words('english')]
    doc_file.close()
    return doc_filtered


# In[259]:

def generate_vocabulary(doc_filtered):
    counter = Counter(doc_filtered)
    extract_docs_tokens = dict()
    i = 0
    for g in counter:
        if 200 <= counter[g]:
            extract_docs_tokens.update({g:i})
            i += 1
    return extract_docs_tokens


# In[260]:

def lsa(tfidf_matrix):
    svd = TruncatedSVD(550, algorithm='arpack', random_state=42)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X=lsa.fit_transform(tfidf_matrix)
    print(svd.explained_variance_ratio_.sum())
    return(X)


# In[1]:

def get_tokens(title):
    tokens = nltk.word_tokenize(title)
    filtered = [w for w in tokens if not w in stopwords.words('english')]
    return filtered

'''def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def get_stem_tokens(title):
    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer("english")
    tokens = nltk.word_tokenize(title)
    filtered = [w for w in tokens if not w in stopwords.words('english')]
    stems = [stemmer.stem(t) for t in filtered]
    stemmed = stem_tokens(filtered, stemmer)
    return stemmed'''


# In[48]:

def read_file(path):
    title_file = open(path, 'r')
    title_content = list()
    for title_line in title_file:
        title_line = title_line.strip()
        lower_line = title_line.lower() 
        translator = str.maketrans({key: ' ' for key in string.punctuation})
        remove_punctuation = lower_line.translate(translator)
        translator = str.maketrans('', '', string.digits)
        remove_digit = remove_punctuation.translate(translator)
        title_content.append(remove_digit)
    title_file.close()
    return(title_content)


# In[83]:

def build_vectorizer(title_content, extract_docs_tokens):
    tfidf_vectorizer = TfidfVectorizer(max_df=1.0, max_features=None, vocabulary=extract_docs_tokens,
                                       min_df=2, stop_words='english', tokenizer=get_tokens,
                                       use_idf=True, ngram_range=(1, 1))
 
    tfidf_matrix = tfidf_vectorizer.fit_transform(title_content) #fit the vectorizer to synopses

    return(tfidf_matrix, tfidf_vectorizer)


# In[141]:

def kmeans(X):
    km = KMeans(n_clusters=20, init='k-means++', max_iter=100, n_init=1, random_state=42)
    km.fit(X)
    clusters = km.labels_.tolist()
    return(clusters, km)


# In[142]:

def mini_batch_kmeans(X):
    km = MiniBatchKMeans(n_clusters=20, init='k-means++', n_init=1,
                     init_size=1000, batch_size=1000, random_state=42)
    km.fit(X)
    clusters = km.labels_.tolist()
    return(clusters, km)


# In[7]:

def write_result(result_path, test_path, clusters):
    result_file = open(result_path, 'w')
    test_file = open(test_path, 'r')
    print('ID', 'Ans', sep=',', file=result_file)
    i = 0
    for test_line in test_file:
        test_line = test_line.strip()
        test_data = test_line.split(',')
        if test_data[0] == 'ID':
            continue
        if clusters[int(test_data[1])] == clusters[int(test_data[2])]:
            print(i, 1, sep=',', file=result_file)
        else:
            print(i, 0, sep=',', file=result_file)
        i += 1 
    test_file.close()
    result_file.close()
    return


# In[8]:

'''------------------------------------------------------------------------------'''

# In[262]:

import string
import nltk
import numpy as np
import os
import sys
import time
import pickle

from collections import Counter
from nltk.corpus import stopwords
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


start_time = time.time()

#nltk.download('punkt')
#nltk.download("stopwords")
#nltk.download('averaged_perceptron_tagger')

folder = sys.argv[1]
#folder = './'

title_path = folder + '/title_StackOverflow.txt'
test_path = folder + '/check_index.csv'
doc_path = folder + '/docs.txt'

result_path = sys.argv[2]
#result_path = 'result.csv'

print('read file...')
title_content = read_file(title_path)

print('read doc.txt...')
#new_doc_filtered = read_doc(doc_path)

#output_tokens = open(folder + '/tokens.p', 'wb')
#pickle.dump(doc_filtered, output_tokens, pickle.HIGHEST_PROTOCOL)
#output_tokens.close()

input_tokens = open('./tokens.p', 'rb')
new_doc_filtered = pickle.load(input_tokens)
input_tokens.close()

print('build vocabulary...')
extract_docs_tokens = generate_vocabulary(new_doc_filtered)

print('build vectorizer...')
tfidf_matrix, vectorizer = build_vectorizer(title_content, extract_docs_tokens)

print('clustering...')
clusters, km = mini_batch_kmeans(lsa(tfidf_matrix))
#clusters, km = kmeans(lsa(tfidf_matrix))

print('write result...')
write_result(result_path, test_path, clusters)
print('done')

print(time.time()-start_time)

# In[69]:

'''doc_file = open('docs.txt', 'r')
doc_content = doc_file.read()
doc_content = doc_content.lower()
translator = str.maketrans({key: ' ' for key in string.punctuation})
doc_no_punctuation = doc_content.translate(translator)
translator = str.maketrans('', '', string.digits)
doc_remove_digit = doc_no_punctuation.translate(translator)
doc_tokens = nltk.word_tokenize(doc_remove_digit)
doc_filtered = [w for w in doc_tokens if not w in stopwords.words('english')]


# In[ ]:

test = list()
title_file = open('title_StackOverflow.txt', 'r')
test_avg = 0
t = tfidf_vectorizer.get_feature_names()
for title_line in title_file:
    test_tokens = nltk.word_tokenize(title_line)
    test_filtered = [w for w in test_tokens if not w in stopwords.words('english')]
    test_count = [w for w in test_filtered if w in t]
    if len(test_count) >= 1:
        test_avg += len(test_count)
        test.append(title_line)
        

title_file.close()
print(len(test))
print(test_avg / len(test))


# #print(tfidf_matrix.shape)
# #print(vectorizer.get_feature_names()[1304])
# 
# test_file = open(test_path, 'r')
# 
# i = 1
# same = 0
# different = 0
# for test_line in test_file:
#     test_line = test_line.strip()
#     test_data = test_line.split(',')
# 
#     if test_data[0] == 'ID':
#         continue
#         
#     if i > 100:
#         break
#     i += 1
#     num_a = int(test_data[1])
#     num_b = int(test_data[2])
#     
#     
#     if clusters[num_a] == clusters[num_b]:
#         same += 1
#         print('same')
#     else:
#         different += 1
#         print('different')
#         
#     print(title_content[num_a])
#     print(title_content[num_b])
#     print('---------------------------------------------------')
#     print(tfidf_matrix[num_a])
#     print('---------------------------------------------------')
#     print(tfidf_matrix[num_b])
#     print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
# print(same, different, sep='\t')
# test_file.close()'''
