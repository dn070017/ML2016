
# coding: utf-8

# In[245]:

def load_train_data(data_path, class_n, instance_n):
    train_path = data_path + '/all_label.p'
    train_raw = pickle.load(open(train_path, 'rb'))
    #train_y = np.zeros(class_n * class_n * instance_n).reshape(class_n * instance_n, class_n)
    train_y = list()

    count = 0

    for _class in range(0, class_n):
        for _idx in range(0, instance_n):
            train_x_instance = np.array(train_raw[_class][_idx]).reshape(1, 3, 32, 32)
            train_x_instance = train_x_instance.astype('float32')
            train_x_instance = train_x_instance / 256
            #train_y[count, _class] = 1
            train_y.append(_class)

            if count == 0:
                train_x = train_x_instance
            else:
                train_x = np.vstack((train_x, train_x_instance))

            count += 1

    #output = open('processed_all_label_x.p', 'wb')
    #pickle.dump(train_x, output, pickle.HIGHEST_PROTOCOL)
    #output = open('processed_all_label_y.p', 'wb')
    #pickle.dump(train_y, output, pickle.HIGHEST_PROTOCOL)

    return train_x, train_y


# In[246]:

def load_unlabel_train_data(data_path):
    unlabel_train_path = data_path + 'all_unlabel.p'
    unlabel_raw = pickle.load(open(unlabel_train_path, 'rb'))

    for _idx in range(0, 45000):
        if _idx % 10000 == 0:
            print('processing unlabel train {}...'.format(_idx))
        unlabel_train_x_instance = np.array(unlabel_raw[_idx]).reshape(1, 3, 32, 32)
        unlabel_train_x_instance = unlabel_train_x_instance.astype('float32')
        unlabel_train_x_instance = unlabel_train_x_instance / 256

        if _idx == 0:
            unlabel_train_x = unlabel_train_x_instance
        else:
            unlabel_train_x = np.vstack((unlabel_train_x, unlabel_train_x_instance))

    #output = open('processed_all_unlabel_x.p', 'wb')
    #pickle.dump(unlabel_train_x, output, pickle.HIGHEST_PROTOCOL)

    return unlabel_train_x


# In[247]:

def load_test_data(data_path):
    test_path = data_path + 'test.p'
    test_raw = pickle.load(open(test_path, 'rb'))

    for _idx in range(0, 10000):
        if _idx % 2500 == 0:
            print('processing test {}...'.format(_idx))
        # print(test_raw['ID'][_idx])
        test_x_instance = np.array(test_raw['data'][_idx]).reshape(1, 3, 32, 32)
        test_x_instance = test_x_instance.astype('float32')
        test_x_instance = test_x_instance / 256

        if _idx == 0:
            test_x = test_x_instance
        else:
            test_x = np.vstack((test_x, test_x_instance))

    #output = open('processed_test_x.p', 'wb')
    #pickle.dump(test_x, output, pickle.HIGHEST_PROTOCOL)

    return test_x


# In[248]:

def build_model(class_n):

    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('sigmoid'))
    model.add(Dense(class_n))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


# In[249]:

def baseline_train(train_x, train_y, class_n, batch, epoch):
    print('start setting dnn model...')
    model = build_model(class_n)

    print('start learning parameters...')
    fit_history = model.fit(train_x, train_y, batch_size=batch, nb_epoch=epoch)
    #model.save('model_002.h5')

    return model


# In[250]:

def test(model, test_x, batch, name):
    result = model.predict_classes(test_x)
    output = open(name, 'w')

    print('ID,class', file=output)
    for i in range(0, len(result)):
        print(str(i), str(result[i]), sep=',', file=output)
    print('analysis complete')

    return


# In[267]:

def unlabel_method1_train(train_x, train_y, unlabel_x, class_n, batch, epoch):
    iteration = 2
    threshold = 0.75

    print('start setting initial dnn model...')
    model = build_model(class_n)

    buffer_train_x = train_x
    buffer_train_y = train_y
    buffer_unlabel_x = unlabel_x

    for i in range(0, iteration):

        print('learning parameters ({})...'.format(i))
        fit_history = model.fit(buffer_train_x, to_categorical(buffer_train_y, 10),
        batch_size=batch, nb_epoch=epoch)

        print('determine unlabel classess ({})...'.format(i))
        unlabel_result = model.predict_classes(buffer_unlabel_x)
        unlabel_result_prob = model.predict_proba(buffer_unlabel_x)

        result_prob_max = np.amax(unlabel_result_prob, axis=1)

        index = np.arange(0, buffer_unlabel_x.shape[0])
        positive_index = index[result_prob_max >= threshold]
        negative_index = index[result_prob_max < threshold]

        if positive_index.shape[0] == 0 or negative_index.shape[0] == 0:
            break

        add_y = np.array(unlabel_result[positive_index])
        #buffer_train_y = np.vstack((buffer_train_y, add_y))
        buffer_train_y = np.append(buffer_train_y, add_y)
        buffer_train_x = np.vstack((buffer_train_x, buffer_unlabel_x[positive_index]))
        buffer_unlabel_x = buffer_unlabel_x[negative_index]


    return model


# In[273]:

def unlabel_method2_train(train_x, train_y, unlabel_x, class_n, batch, epoch):
    print('encoding...')
    encoding_dim = 32

    input_img = Input(shape=(3072,))
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    decoded = Dense(3072, activation='sigmoid')(encoded)

    autoencoder = Model(input=input_img, output=decoded)
    encoder = Model(input=input_img, output=encoded)

    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    t_train_x = train_x.reshape(2500, 3072)
    t_unlabel_x = unlabel_x.reshape(45000, 3072)

    autoencoder.fit(t_train_x, t_train_x,
                    nb_epoch=20,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(t_test_x, t_test_x))

    train_code = encoder.predict(t_train_x)
    unlabel_code = encoder.predict(t_unlabel_x)

    print('predict unlabel data with SVM...')
    clf = svm.SVC()
    clf.fit(train_code, train_y)
    unlabel_predict = clf.predict(unlabel_code)

    buffer_train_y = np.append(train_y, unlabel_predict)
    buffer_train_x = np.vstack((train_x, unlabel_x))
    model = baseline_train(buffer_train_x, to_categorical(buffer_train_y, 10), class_n,
            batch, epoch)

    return model


# In[276]:

def cross_validation(train_x, train_y, unlabel_train_x, class_n, batch, epoch):

    np.random.seed(seed=0)

    index = np.arange(0, 5000)
    np.random.shuffle(index)

    train_index = index[0:2500]
    test_index = index[2500:5000]

    buffer_train_x = train_x[train_index]
    buffer_test_x = train_x[test_index]

    buffer_train_y = train_y[train_index]
    buffer_test_y = train_y[test_index]

    #model = baseline_train(buffer_train_x, to_categorical(buffer_train_y, 10), class_n, batch, epoch)
    #result, result_prob, model = unlabel_method1_train(buffer_train_x, buffer_train_y, unlabel_train_x, class_n, batch, epoch)
    model = unlabel_method2_train(buffer_train_x, buffer_train_y, unlabel_train_x, class_n, batch, epoch)
    fit_evaluation_1 = model.evaluate(buffer_test_x, to_categorical(buffer_test_y, 10), batch_size=batch)

    #model = baseline_train(buffer_test_x, to_categorical(buffer_test_y, 10), class_n, batch, epoch)
    #result, result_prob, model = unlabel_method1_train(buffer_test_x, buffer_test_y, unlabel_train_x, class_n, batch, epoch)
    model = unlabel_method2_train(buffer_test_x, buffer_test_y, unlabel_train_x, class_n, batch, epoch)
    fit_evaluation_2 = model.evaluate(buffer_train_x, to_categorical(buffer_train_y, 10), batch_size=batch)

    print(fit_evaluation_1[1], fit_evaluation_2[1], sep='\t')

    return


# In[277]:

import sys
import numpy as np
import pickle

from sklearn import svm

from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical

K.set_image_dim_ordering('th')

class_n = 10
instance_n = 500
batch = 100
epoch = 20

data_path = sys.argv[1]
model_path = sys.argv[2]
#result_path = sys.argv[3]

print('start loading training set...')
train_x, train_y = load_train_data(data_path, class_n, instance_n)
#train_x = pickle.load(open('processed_all_label_x.p', 'rb'))
#train_y = pickle.load(open('processed_all_label_y.p', 'rb'))
original_y = np.array(train_y).astype(int)
train_y = to_categorical(train_y, class_n)

print('start reading unlabel training set...')
unlabel_train_x = load_unlabel_train_data(data_path)
#unlabel_train_x = pickle.load(open('processed_all_unlabel_x.p', 'rb'))

print('start loading testing set...')
#test_x = load_test_data(data_path)
#test_x = pickle.load(open('processed_test_x.p', 'rb'))

#print('cross validation')
#cross_validation(train_x, original_y, unlabel_train_x, class_n, batch, epoch)

print('training baseline model')
#model = baseline_train(train_x, train_y, class_n, batch, epoch)
#test(model, test_x, epoch, 'result_final.csv')
#model.save('model_final.h5')

print('unsupervise training baseline model (method 1)')
model = unlabel_method1_train(train_x, original_y, unlabel_train_x, class_n, batch, epoch)
#test(model, test_x, epoch, 'usm1_result_final.csv')
model.save(model_path)

print('unsupervise training baseline model (method 2)')
#model = unlabel_method2_train(train_x, original_y, unlabel_train_x, class_n, batch, epoch)
#test(model, test_x, epoch, 'usm2_result_final.csv')
#model.save(model_path)


# In[ ]:
