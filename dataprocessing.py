import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

label_name_file = '/home/qjq/data/NUS-WIDE/ConceptsList/Concepts81.txt'

# 处理label
label_pos = {}
pos_label = {}
with open(label_name_file, mode='r') as f:
    for i, line in enumerate(f):
        label_pos[line[:-1]] = i
        pos_label[i] = line[: -1]
label_class = len(label_pos)


# read image file list
image_list_file = '/home/qjq/data/NUS-WIDE/ImageList/Imagelist.txt'
image_dir = '/home/qjq/data/NUS-WIDE/Flickr/'
image_list = None
with open(image_list_file, mode='r') as f:
    image_list = [image_dir + line[: -1] for line in f]


label_matrix = np.zeros((len(image_list), label_class), dtype='uint8')
label_count = []
label_file_dir = '/home/qjq/data/NUS-WIDE/Groundtruth/AllLabels/'
label_file_list = os.listdir(label_file_dir)

for label_file in label_file_list:
    # an example of the filename: Labels_airport.txt
    # remove the prefix "Labels_" and the suffix ".txt"
    current_label_name = label_file[7: -4]
    current_label_index = label_pos[current_label_name]

    current_count = 0
    with open(label_file_dir + label_file, mode='r') as f:
        for current_image_index, line in enumerate(f):
            label_matrix[current_image_index][current_label_index] = int(line[: -1])
            current_count += int(line[: -1])
    label_count.append((current_label_name, current_count))

# sort the label count in decreasing order
# then we can pickup the top k frequent labels
label_count.sort(key=lambda x: -x[1])
top_k = 21
top_k_label = [name for name, _ in label_count[: top_k]]
top_k_one_hot = np.zeros(label_class, dtype='uint8')

for name, _ in label_count[: top_k]:
    top_k_one_hot[label_pos[name]] = 1


# shuffle the data
shuffle_index = np.random.choice(len(image_list), size=len(image_list), replace=False)
image_list = [image_list[index] for index in shuffle_index]
label_matrix = np.array([label_matrix[index] for index in shuffle_index])

image_list_top_k_category = [img for i, img in enumerate(image_list) if np.dot(label_matrix[i], top_k_one_hot) > 0]
label_matrix_top_k_category = [label for label in label_matrix if np.dot(label, top_k_one_hot) > 0]

image_list = image_list_top_k_category
label_matrix = label_matrix_top_k_category

TRAIN_SIZE = 500
QUERY_SIZE = 100

train_index = {}
query_index = {}
database_index = []

for name in top_k_label:
    train_index.setdefault(name, [])
    query_index.setdefault(name, [])

for index in range(len(image_list)):
    selected = False
    in_top_k = False
    for pos, bit in enumerate(label_matrix[index]):
        if bit and top_k_one_hot[pos]:
            in_top_k = True
            cur_name = pos_label[pos]
            if len(train_index[cur_name]) < TRAIN_SIZE:
                train_index[cur_name].append(index)
                database_index.append(index)
                selected = True
                break
            elif len(query_index[cur_name]) < QUERY_SIZE:
                query_index[cur_name].append(index)
                selected = True
                break
    if not selected and in_top_k:
        database_index.append(index)

train_index = [index for _, index_arr in train_index.items() for index in index_arr]
query_index = [index for _, index_arr in query_index.items() for index in index_arr]

print('trainint set:', len(train_index))
print('query set:', len(query_index))
print('database:', len(database_index))

x_query = [image_list[index] for index in query_index]
x_train = [image_list[index] for index in train_index]
x_database = [image_list[index] for index in database_index]

y_query = np.array([label_matrix[index] for index in query_index])
y_train = np.array([label_matrix[index] for index in train_index])
y_database = np.array([label_matrix[index] for index in database_index])


dataset = [
    (x_train, y_train, 'train'),
    (x_query, y_query, 'query'),
    (x_database, y_database, 'database')
]
for x_dataset, y_dataset, name in dataset:
    print('Processing ' + name + ' data')

    os.system('mkdir -p ' + '/home/qjq/project/MyDPSH_nus/nus_wide_21/' + name + '/image/')
    with open('/home/qjq/project/MyDPSH_nus/nus_wide_21/' + name + '/image/' + 'image_list.txt', mode='w') as f:
        for image_path in x_dataset:
            f.write(image_path + '\n')
    os.system('mkdir -p ' + '/home/qjq/project/MyDPSH_nus/nus_wide_21/' + name + '/label/')
    np.save('/home/qjq/project/MyDPSH_nus/nus_wide_21/' + name + '/label/' + 'label.npy', y_dataset)
    print('Processing finished')

