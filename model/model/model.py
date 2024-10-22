# 1
import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout
from keras.optimizers import Adam

from tensorflow.keras.backend import ctc_batch_cost

# 2
train = pd.read_csv('C:\\Users\\purus\\Desktop\\modelService\\dataset\\written_name_train_v2.csv\\written_name_train_v2.csv')
valid = pd.read_csv('C:\\Users\\purus\\Desktop\\modelService\\dataset\\written_name_validation_v2.csv\\written_name_validation_v2.csv')

# 3
train
# print(train)

# 4
valid
# print(valid)

# 5
plt.figure(figsize=(15, 10))

for i in range(6):
    ax = plt.subplot(2, 3, i+1)
    img_dir = 'C:\\Users\\purus\\Desktop\\modelService\\dataset\\train_v2\\train\\'+ train.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap = 'gray')
    plt.title(train.loc[i, 'IDENTITY'], fontsize=12)
    plt.axis('off')

plt.subplots_adjust(wspace=0.2, hspace=-0.8)

# 6
train.info()

# 7
valid.info()

# 8
print("Number of NaNs in train set      : ", train['IDENTITY'].isnull().sum())
print("Number of NaNs in validation set : ", valid['IDENTITY'].isnull().sum())

# 9
train.dropna(axis=0, inplace=True)
valid.dropna(axis=0, inplace=True)

# 10
unreadable = train[train['IDENTITY'] == 'UNREADABLE']
unreadable.reset_index(inplace = True, drop=True)

plt.figure(figsize=(15, 10))

for i in range(6):
    ax = plt.subplot(2, 3, i+1)
    img_dir = 'C:\\Users\\purus\\Desktop\\modelService\\dataset\\train_v2\\train\\'+unreadable.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap = 'gray')
    plt.title(unreadable.loc[i, 'IDENTITY'], fontsize=12)
    plt.axis('off')

plt.subplots_adjust(wspace=0.2, hspace=-0.8)

# 11
train = train[train['IDENTITY'] != 'UNREADABLE']
valid = valid[valid['IDENTITY'] != 'UNREADABLE']

# 12
train['IDENTITY'] = train['IDENTITY'].str.upper()
valid['IDENTITY'] = valid['IDENTITY'].str.upper()

# 13
train.reset_index(inplace = True, drop=True) 
valid.reset_index(inplace = True, drop=True)

# 14
def preprocess(img):
    (h, w) = img.shape
    
    final_img = np.ones([64, 256])*255 
    
    # crop
    if w > 256:
        img = img[:, :256]
        
    if h > 64:
        img = img[:64, :]
    
    
    final_img[:h, :w] = img
    return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)

# 15
train_size = 30000
valid_size= 3000

# 16
train_x = []

for i in range(train_size):
    img_dir = 'C:\\Users\\purus\\Desktop\\modelService\\dataset\\train_v2\\train\\'+train.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    image = preprocess(image)
    image = image/255.
    train_x.append(image)

# 17
valid_x = []

for i in range(valid_size):
    img_dir = 'C:\\Users\\purus\\Desktop\\modelService\\dataset\\validation_v2\\validation\\'+valid.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    image = preprocess(image)
    image = image/255.
    valid_x.append(image)    

# 18
train_x = np.array(train_x).reshape(-1, 256, 64, 1)
valid_x = np.array(valid_x).reshape(-1, 256, 64, 1)    

# 19
alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
max_str_len = 24 # max length of input labels
num_of_characters = len(alphabets) + 1 # +1 for ctc pseudo blank
num_of_timestamps = 64 # max length of predicted labels

def label_to_num(label):
    label_num = []
    for ch in label:
        label_num.append(alphabets.find(ch))
        
    return np.array(label_num)

def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret+=alphabets[ch]
    return ret

# 20
name = 'NUR HIKMAH'
print(name, '\n',label_to_num(name))  

# 21
train_y = np.ones([train_size, max_str_len]) * -1
train_label_len = np.zeros([train_size, 1])
train_input_len = np.ones([train_size, 1]) * (num_of_timestamps-2)
train_output = np.zeros([train_size])

for i in range(train_size):
    train_label_len[i] = len(train.loc[i, 'IDENTITY'])
    train_y[i, 0:len(train.loc[i, 'IDENTITY'])]= label_to_num(train.loc[i, 'IDENTITY'])    


# 22
valid_y = np.ones([valid_size, max_str_len]) * -1
valid_label_len = np.zeros([valid_size, 1])
valid_input_len = np.ones([valid_size, 1]) * (num_of_timestamps-2)
valid_output = np.zeros([valid_size])

for i in range(valid_size):
    valid_label_len[i] = len(valid.loc[i, 'IDENTITY'])
    valid_y[i, 0:len(valid.loc[i, 'IDENTITY'])]= label_to_num(valid.loc[i, 'IDENTITY'])  

# 23
print('True label : ',train.loc[100, 'IDENTITY'] , '\ntrain_y : ',train_y[100],'\ntrain_label_len : ',train_label_len[100], 
      '\ntrain_input_len : ', train_input_len[100])


# 24
# Input layer: expects grayscale images of shape (256, 64, 1)
input_data = Input(shape=(256, 64, 1), name='input')

# First Convolutional Block
inner = Conv2D(32, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_data)  
inner = BatchNormalization()(inner)  
inner = Activation('relu')(inner) 
inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)  

# Second Convolutional Block
inner = Conv2D(64, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)  #
inner = BatchNormalization()(inner) 
inner = Activation('relu')(inner)  
inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner) 
inner = Dropout(0.3)(inner)  

# Third Convolutional Block
inner = Conv2D(128, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)  # 128 filters, 3x3 kernel
inner = BatchNormalization()(inner) 
inner = Activation('relu')(inner)  
inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)  
inner = Dropout(0.3)(inner) 

# Reshape layer to prepare for RNN input
inner = Reshape(target_shape=((64, 1024)), name='reshape')(inner)  # Reshape to (64, 1024)
inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner) 

# Recurrent Neural Network (RNN) Layers
inner = Bidirectional(LSTM(256, return_sequences=True), name='lstm1')(inner)  # Bidirectional LSTM with 256 units, returns sequences
inner = Bidirectional(LSTM(256, return_sequences=True), name='lstm2')(inner)  # Another Bidirectional LSTM with 256 units

# Output layer
inner = Dense(num_of_characters, kernel_initializer='he_normal', name='dense2')(inner) 
y_pred = Activation('softmax', name='softmax')(inner)  

# Model definition
model = Model(inputs=input_data, outputs=y_pred)  
model.summary() 

# 25    ---- CTC LOST ----  
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

# def ctc_lambda_func(args):
#     y_pred, labels, input_length, label_length = args
#     y_pred = y_pred[:, 2:, :]  # Ignore the first two time steps
#     return ctc_batch_cost(labels, y_pred, input_length, label_length)

# 26
labels = Input(name='gtruth_labels', shape=[max_str_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
model_final = Model(inputs=[input_data, labels, input_length, label_length], outputs=ctc_loss)


# 27 ----   COMPILE MODEL   ----
# the loss calculation occurs elsewhere, so we use a dummy lambda function for the loss

model_final.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(learning_rate=0.0001))

# 28    ---- MODEL TRAINING ----
history = model_final.fit(x=[train_x, train_y, train_input_len, train_label_len], y=train_output, 
                          validation_data=([valid_x, valid_y, valid_input_len, valid_label_len], valid_output),
                          epochs=60, batch_size=128)


# 29
# Function to plot the training history
def plot_training_history(history):
    plt.figure(figsize=(8,5))

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

# Plot the training history
plot_training_history(history)

# 30
preds = model.predict(valid_x)
decoded = K.get_value(K.ctc_decode(preds, input_length=np.ones(preds.shape[0])*preds.shape[1], 
                                   greedy=True)[0][0])

prediction = []
for i in range(valid_size):
    prediction.append(num_to_label(decoded[i]))


# Assuming 'preds' is the output from your model
# preds = ...  # Your model predictions
# input_length = tf.ones(preds.shape[0]) * preds.shape[1]  # Modify as per your actual input lengths

# Example code snippet
# preds = model.predict(valid_x)  # Get the model predictions

# Ensure preds has the expected shape before calculating input_length
# if preds.ndim == 3:  # Ensure preds is a 3D tensor
#     input_length = tf.ones(preds.shape[0]) * preds.shape[1]
# else:
#     raise ValueError("Predictions shape is not as expected.")


# # Decode the predictions
# decoded, log_probabilities = tf.nn.ctc_decode(preds, input_length=input_length, greedy=True)

# # If you need to access the decoded result as numpy
# decoded_numpy = decoded.numpy()  # Convert tensor to numpy array if needed

# 31
y_true = valid.loc[0:valid_size, 'IDENTITY']
correct_char = 0
total_char = 0
correct = 0

for i in range(valid_size):
    pr = prediction[i]
    tr = y_true[i]
    total_char += len(tr)
    
    for j in range(min(len(tr), len(pr))):
        if tr[j] == pr[j]:
            correct_char += 1
            
    if pr == tr :
        correct += 1 
    
print('Correct characters predicted : %.2f%%' %(correct_char*100/total_char))
print('Correct words predicted      : %.2f%%' %(correct*100/valid_size)) 

# 32
# Calculate accuracy based on character and word level
correct_char_percent = correct_char * 100 / total_char
correct_word_percent = correct * 100 / valid_size

# Plotting the accuracy with custom colors
plt.figure(figsize=(8, 6))
plt.bar(['Character Accuracy', 'Word Accuracy'], 
        [correct_char_percent, correct_word_percent], 
        color=['skyblue', 'lightgreen'])  # Ganti warna batang di sini
plt.ylim(0, 100)
plt.title('Model Accuracy Evaluation')
plt.ylabel('Accuracy (%)')
plt.show()

# 33
def calculate_cer(gt_texts, pred_texts):
    """
    Calculate the Character Error Rate (CER) between ground truth texts and predicted texts.
    """
    total_errors = 0
    total_chars = 0
    
    for gt, pred in zip(gt_texts, pred_texts):
        total_errors += edit_distance(gt, pred)
        total_chars += len(gt)
    
    cer = total_errors / total_chars
    return cer

def calculate_wer(gt_texts, pred_texts):
    """
    Calculate the Word Error Rate (WER) between ground truth texts and predicted texts.
    """
    total_errors = 0
    total_words = 0
    
    for gt, pred in zip(gt_texts, pred_texts):
        total_errors += edit_distance(gt.split(), pred.split())
        total_words += len(gt.split())
    
    wer = total_errors / total_words
    return wer

def edit_distance(s1, s2):
    """
    Calculate the Levenshtein distance between two strings.
    """
    if len(s1) < len(s2):
        return edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

ground_truths = valid.loc[:valid_size, 'IDENTITY'].tolist()
predictions = prediction

cer = calculate_cer(ground_truths, predictions)
wer = calculate_wer(ground_truths, predictions)

print('Character Error Rate (CER): {:.2f}%'.format(cer * 100))
print('Word Error Rate (WER): {:.2f}%'.format(wer * 100))

# 34
cer_values = [cer]  # CER value
wer_values = [wer]  # WER value
labels = ['CER', 'WER']

# Plotting
plt.figure(figsize=(8, 6))
plt.bar(labels, [cer * 100, wer * 100], color=['blue', 'green'])
plt.ylim(0, 100)  # Set y-axis limit from 0% to 100%
plt.ylabel('Error Rate (%)')
plt.title('Character Error Rate (CER) vs Word Error Rate (WER)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate([cer * 100, wer * 100]):
    plt.text(i, v + 2, f'{v:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.show()

# 35
test = pd.read_csv('C:\\Users\\purus\\Desktop\\modelService\\dataset\\written_name_test_v2.csv')

plt.figure(figsize=(15, 10))
for i in range(6):
    ax = plt.subplot(2, 3, i+1)
    img_dir = 'C:\\Users\\purus\\Desktop\\modelService\\dataset\\test_v2\\test\\'+test.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap='gray')
    
    image = preprocess(image)
    image = image/255.
    pred = model.predict(image.reshape(1, 256, 64, 1))
    decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], 
                                       greedy=True)[0][0])
    plt.title(num_to_label(decoded[0]), fontsize=12)
    plt.axis('off')
    
plt.subplots_adjust(wspace=0.2, hspace=-0.8)

# 36
model_final.save('model_upto_50.h5')
