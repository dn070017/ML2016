
# coding: utf-8

# In[70]:

line_counter = 0
new_id = 0
feature_colnames = ''
feature_rownames = list()
buffer_feature_data = list()

original_feature_file = open('train.csv', 'r', encoding='Big5')
processed_feature_file = open('processed_train.csv', 'w', encoding='Big5')

for original_feature_line in original_feature_file:
    
    line_counter += 1
    
    original_feature_line = original_feature_line.strip()
    original_feature_data = original_feature_line.split(',')

    if original_feature_data[2] == 'RAINFALL':
        continue
    if line_counter == 1:
        feature_colnames = original_feature_data
        continue
    if line_counter <= 19:
        feature_rownames.append(original_feature_data[2])
    
    buffer_feature_data.append(original_feature_data[3:])
       
    # print(str(len(buffer_feature_data)) + '\t' + str(len(feature_rownames)))
    
    if original_feature_data[2] == 'WS_HR':
        for i in range(0, 15):
            for j in range(0, len(buffer_feature_data)):
                new_id += 1
                print('id_'+ str(new_id) + ',' + feature_rownames[j] + ',', end='', file=processed_feature_file)
                print(*buffer_feature_data[j][i:i+9], sep=',', file=processed_feature_file)
        buffer_feature_data.clear()
    
    # print(*buffer_feature_data, sep='\t')

# print(*feature_rownames, sep='\t')

original_feature_file.close()
processed_feature_file.close()


# In[47]:

a = [1, 2, 3]
b = [4, 5, 6]
c.clear()
c.append(a)
c.append(b)
for i in range(0, 2):
    for tt in c:
        print(*tt[i:i+2], sep='\t')


# In[ ]:



