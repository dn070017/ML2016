
# coding: utf-8

# In[507]:

def train_main():
    train_meta = np.genfromtxt(sys.argv[1], delimiter=',')
    train_id = train_meta[:,0].astype(int)
    train_label = train_meta[:,-1]
    train_feature = train_meta[:,1:-1]
    
    train_feature = normalize(train_feature)
    
    bias = 1
    
    weight = logistic_regression(train_feature, train_label, bias)
    
    np.savetxt(sys.argv[2], weight, delimiter=',')
    
    return


# In[513]:

def test_main():
    test_meta = np.genfromtxt(sys.argv[2], delimiter=',')
    test_id = test_meta[:,0].astype(int)
    test_feature = test_meta[:,1:]

    test_feature = normalize(test_feature)
    
    bias = 1
    weight = np.genfromtxt(sys.argv[1], delimiter=',')
    y_prob = calc_sigmoid_function(test_feature, weight, bias)
    y_pred = classify(y_prob).astype(int)
    
    output_file = open(sys.argv[3], 'w')
    print('id,label', file=output_file)
    for i in range(0, test_feature.shape[0]):
        print(i+1, y_pred[i], sep=',', file=output_file)
        #print(i+1, y_pred[i], sep=',')
    output_file.close()


# In[514]:

def normalize(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)


# In[515]:

def logistic_regression(train_feature, train_label, bias):
    weight = np.ones(train_feature.shape[1])
    
    learning_rate = 0.001
    iteration = 10000
    
    for i in range(0, iteration):
        y_prob = calc_sigmoid_function(train_feature, weight, bias)
        
        gradient = -1 * np.dot(train_label - y_prob, train_feature)
        
        weight = weight - learning_rate * gradient
             
        y_prob = calc_sigmoid_function(train_feature, weight, bias)
        y_pred = classify(y_prob)
        accuracy = np.where(y_pred == train_label)
        #if i % 1000 == 0:
        #    print(len(accuracy[0]))
        
    return weight


# In[516]:

def calc_sigmoid_function(train_feature, w, b):
    z = np.dot(w, np.transpose(train_feature)) + b
    return 1 / (1 + np.exp(-1 * z))


# In[517]:

def classify(prob):
    pred = copy.copy(prob)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    return pred


# In[518]:

import copy
import math
import numpy as np
import sys

train_main()
#test_main()

