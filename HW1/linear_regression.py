def main():

    # parsing training feature data
    feature_dict = defaultdict(list)
    feature_file = open('processed_train_0_n.csv', 'r')
    for feature_line in feature_file:
        feature_line = feature_line.strip()
        temp_feature_data = feature_line.split(',')
        feature_dict[temp_feature_data[1]].append(temp_feature_data)
    feature_file.close()

    # parsing training label data
    label_data = list()
    label_file = open('processed_label_0_n.csv', 'r')
    label_header = label_file.readline()
    for label_line in label_file:
        label_line = label_line.strip()
        label_data.append(label_line.split(','))
    label_file.close()

    # parsing testing data
    test_dict = defaultdict(list)
    test_file = open('test_X.csv', 'r')
    for test_line in test_file:
        test_line = test_line.strip()
        temp_test_data = test_line.split(',')
        if temp_test_data[1] == 'RAINFALL':
            continue
        test_dict[temp_test_data[1]].append(temp_test_data)

    # n-fold cross validation
    '''fold = 4
    testing_rmse = list()
    indices = np.random.permutation(len(label_data))
    indices_count = indices.shape[0]
    partition_size = indices_count // fold
    for i in range(0, fold):
        test_start = partition_size * i
        test_end = partition_size * (i + 1) - 1
        test_id = indices[test_start:test_end]
        print('{} fold ({} indices):'.format(i + 1, test_end - test_start))
        for j in range(0, fold):
            train_start = partition_size * j
            train_end = partition_size * (j + 1) - 1
            train_id = indices[train_start:train_end]
            if train_start == test_start and train_end == test_end:
                continue
            else:
                result = linear_regression(feature_dict, label_data, feature_dict, train_id, test_id)
                label_array = np.array(label_data)[test_id, 1]
                label_array = label_array.astype('float64')
                rmse = np.sqrt(np.mean(np.power(np.subtract(label_array, result), 2)))
                testing_rmse.append(rmse)
                print('rmse for testing set: {:.5f}'.format(rmse))
                #plt.scatter(result, label_array)'''

    output_file = open('linear_regression.csv', 'w')

    #print('final model ({} indices):'.format(len(label_data)))
    result = linear_regression(feature_dict, label_data, test_dict, np.arange(len(label_data)),                                np.arange(len(test_dict['PM2.5'])))
    #print('rmse for {} fold cross-validation: {:.5f}'.format(fold, sum(testing_rmse) / float(len(testing_rmse))))
    #print(result)
    print('id,value', file=output_file)
    for i in range(0, result.shape[0]):
        print('id_' + str(i) + ',', end='', file=output_file)
        if result[i] < -1:
            result[i] = -1
        print(result[i], file=output_file)
    output_file.close()


# In[130]:

def linear_regression(feature_dict, label_data, test_dict, train_id, test_id):

    # convert label data to label array
    label_array = np.array(label_data)[train_id, 1]
    label_array = label_array.astype('float64')

    # only consider features in field_set
    field_set = {'PM2.5', 'PM10', 'NO2', 'NOx'}

    # construct feature array for training and testing data
    count = 0
    for field, data_array in feature_dict.items():
        if field not in field_set:
            continue
        train_field_feature_array = np.array(data_array)[train_id,2:]
        train_field_feature_array = train_field_feature_array.astype('float64')
        test_field_feature_array = np.array(test_dict[field])[test_id,2:]
        test_field_feature_array = test_field_feature_array.astype('float64')

        # time series (-0 ~ -9)
        # consider only one degree polynomial formula
        for i in range(0, 9):
            sub_train_feature_array = train_field_feature_array[:,i]
            sub_test_feature_array = test_field_feature_array[:,i]
            if count == 0:
                count += 1
                train_feature_array = sub_train_feature_array
                test_feature_array = sub_test_feature_array
            else:
                train_feature_array = np.vstack((train_feature_array, sub_train_feature_array))
                test_feature_array = np.vstack((test_feature_array, sub_test_feature_array))

    # add pseudo variant (w0 always 1)
    pseudo_var = np.ones(train_feature_array.shape[1])
    train_feature_array = np.vstack((pseudo_var, train_feature_array))
    pseudo_var = np.ones(test_feature_array.shape[1])
    test_feature_array = np.vstack((pseudo_var, test_feature_array))

    # scale data to mean 0, standard deviation: 1
    for i in range(1, train_feature_array.shape[0]):
        train_standard_deviation = np.std(train_feature_array[i,])
        test_standard_deviation = np.std(test_feature_array[i,])
        if train_standard_deviation == 0:
            train_standard_deviation += 0.0001
        if test_standard_deviation == 0:
            test_standard_deviation += 0.0001
        train_feature_array[i,] = np.subtract(train_feature_array[i,], \
                                  np.mean(train_feature_array[i,])) / train_standard_deviation
        test_feature_array[i,] = np.subtract(test_feature_array[i,],
                                 np.mean(test_feature_array[i,])) / test_standard_deviation \

    # call gradient descent algorithms
    coefficient_array = adaptive_gradient_descent(train_feature_array, label_array)
    prediction = coefficient_array.dot(test_feature_array)
    return prediction


# In[131]:

def adaptive_gradient_descent(target_array, label_array):
    # setting parameters
    learning_rate = 50
    regular_lambda = 0.001
    iteration = 50000

    feature_count = target_array.shape[0]
    instance_count = target_array.shape[1]

    coefficient_array = np.ones(feature_count)
    gradient_w = np.ones(feature_count)
    old_gradient_w = np.zeros((feature_count, iteration))

    last_error = sys.maxsize
    current_error = sys.maxsize

    for i in range(0, iteration):
        for j in range(0, feature_count):
            # gradient_descent
            gradient_w[j] = -1 / instance_count * np.dot(np.subtract(label_array, \
                            coefficient_array.dot(target_array)), np.transpose(target_array[j,:])) + \
                            regular_lambda * coefficient_array[j]

            # store old_gradient_w (for adaptive gradient descent)
            if i == 0:
                old_gradient_w[j, i] = gradient_w[j] ** 2
            else:
                old_gradient_w[j, i] = old_gradient_w[j, i-1] + gradient_w[j] ** 2

        for j in range(0, feature_count):
            # update coefficient_array theta
            coefficient_array[j] -= learning_rate / math.sqrt(old_gradient_w[j, i]) * gradient_w[j]

        current_error = np.sqrt(np.mean(np.power(np.subtract(label_array, \
                                coefficient_array.dot(target_array)), 2)))
        # stop iteration if the change of error is smaller than 1e-13
        if i >= iteration * 0.5:
            if last_error - current_error <= 1e-13:
                break

        last_error = current_error

    #print('rmse for training set: {:.5f}'.format(current_error), end='\t')
    return(coefficient_array)

# In[132]:

from collections import defaultdict

import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import time

start = time.time()
np.random.seed(1)
main()
#print('training time: {:.3f} seconds'.format(time.time() - start))

# if __name__ == 'main':
#     main()

