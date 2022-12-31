from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import re
import os
import sys
import json
import time
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import matplotlib.pyplot as plt

LOWEST_LOSS = 0

def gather_text():
    text = []
    clean_text = []

    with open('cheat_sheet.txt', 'r', encoding = 'utf-8') as r:
        text.append(r.read())
    
    for i in text:
        x = i.strip()
        x = x.replace('\n', '')
        clean_text.append(x)
        
    return clean_text
        
def make_data():
    text = []
    clean_text = []
    retweet = []
    labels = []

    def gather_data(year):
        if year == 2019:
            current_dir = "condensed_" + str(year) + ".json" + "/condensed_" + str(year) + ".json"
            with open(current_dir, 'r', encoding = 'utf-8') as f:
                data = json.load(f)
                print('opening: ' + current_dir)

            for item in data:
#                print(item)
                #text.append(item['text'])
                text.append(item['text'] + ' [N]')
#                print(text)
                retweet.append(item['is_retweet'])
#                print(retweet)
#                labels.append(0)
#                print(labels)
#                break
    
        else:    
            for i in range(10):
                current_dir = "condensed_" + str(year) + ".json" + "/condensed_" + str(year) + ".json"
                with open(current_dir, 'r', encoding = 'utf-8') as f:
                    data = json.load(f)
                    print('opening: ' + current_dir)
                    
                for item in data:
#                    print(item)
                    #text.append(item['text'])
                    text.append(item['text'] + ' [N]')
#                    print(text)
                    retweet.append(item['is_retweet'])
#                    print(retweet)
#                    labels.append(0)
#                    print(labels)
#                    break
        
                year += 1

    def other_data(name):
        current_dir = "Others/" + name + ".json"
        with open(current_dir, 'r', encoding = 'utf-8') as f:
            data = json.load(f)
            print(name)
            
            for item in data:
                text.append(item['text'])
                retweet.append('is_retweet')
                labels.append(1)
                
    gather_data(2009)        
    gather_data(2019)
    #other_data('Ivanka')
    #other_data('Mitch')
    #other_data('Pence')
    #other_data('Stone')

    def english(s):
        #use the ascii encoding to check for english chars
        try:
            s.encode(encoding = 'utf-8').decode('ascii')
        except UnicodeDecodeError:
            return False
        else:
            return True
        
    
    at_pattern = re.compile(r'@(\s)?\w+')
    exact_pattern = re.compile(r'("\s")?(:\s\s?)?(\s?\s:)?(https?://?t?.?)?(https?:?)?(http://nixonssecrets\sdot\scom)?(http://GaryJohnson2012)?(RT\s\s?_?:)?([rR][eE]:)?(&amp;)?')
    website_pattern = re.compile(r'(https?)?(:)?(//)?(www\.)?(\w+)?(\.\w+\/?)(\/)?(\w+\/?)?(\/)?(\w+\/?)?(\/)?(\w+\/?)?(\/)?(\w+\/?)?(\/)?(\w+\/?)?')

    matches = []
    matches_2 = []
    matches_3 = []

    for i in text:
        x = at_pattern.findall(i)
        
        if len(x) != 0:
            matches.append(x)
        
        y = website_pattern.findall(i)
    
        if len(y) != 0:
            matches_2.append(y)
            
        z = exact_pattern.findall(i)
    
        if len(z) != 0:
            matches_3.append(z)
            
    
    for i in text:
        x = website_pattern.sub('', i)
        x = at_pattern.sub('', x)
        x = exact_pattern.sub('', x)
        x = x.strip()
        x = x.replace('\n', ' ')
        
        if len(x) != 0:
            clean_text.append(x)
    

    current_text = ""
    cleanest_text = ""
    
    for i in clean_text:
        for x in i:
            z = english(x)
            
            if z == True:
                current_text += x
        cleanest_text += " "
        cleanest_text += current_text
        current_text = ""
                
#    for num in range(len(clean_text)):
#        print(num)
#        with open('Ivanka Clean.txt', 'a', encoding = 'utf-8') as w:
#            w.write(f"{num}:{clean_text[num]}\n")
#            num += 1
    with open('text.txt', 'a', encoding = 'utf-8') as w:
        w.write(cleanest_text)

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                      batch_input_shape = [batch_size, None]),
            tf.keras.layers.GRU(rnn_units,
                                return_sequences = True,
                                stateful = True,
                                recurrent_initializer = 'glorot_uniform'),
#            tf.keras.layers.LSTM(rnn_units, return_sequences = True, recurrent_initializer = 'glorot_uniform'),
            tf.keras.layers.Dense(vocab_size)
            ])
    return model
    
class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, loss):
        super(CustomCallback, self).__init__()
        self.lowest_loss = loss
        self.cp
        
    def on_epoch_end(self, epoch, logs = None):
        current_loss = logs.get("loss")
        
        if current_loss < self.lowest_loss: 
            tf.keras.callbacks.ModelCheckpoint(self.cp, save_weights_only = True, verbose = 1)
            LOWEST_LOSS = current_loss
        
def AI_stuff(train = True):
    with open('text.txt', 'r', encoding = 'utf-8') as r:
        text = r.read()
        
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath = checkpoint_prefix,
            save_weights_only = True)
        
    vocab = sorted(set(text))
    
    print(vocab)

    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    text_as_int = np.array([char2idx[c] for c in text])

    seq_length = 210

    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    
    sequences = char_dataset.batch(seq_length + 1, drop_remainder = True)
    
    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset = sequences.map(split_input_target)
        
    print(dataset)
    vocab_size = len(vocab)
    
    embedding_dim = 128
    
    rnn_units = 1024
    
    if train:
        lowest_loss = 1000
        BATCH_SIZE = 64

        BUFFER_SIZE = 100000

        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder = True)

        model = build_model(
                vocab_size = len(vocab),
                embedding_dim = embedding_dim,
                rnn_units = rnn_units,
                batch_size = BATCH_SIZE)

        for input_example_batch, target_example_batch in dataset.take(1):
            example_batch_predictions = model(input_example_batch)

        sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples = 1)
        sampled_indices = tf.squeeze(sampled_indices, axis =-1).numpy()

        def loss(labels, logits):
            return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits = True)

        model.compile(optimizer = tf.keras.optimizers.Adam(), loss = loss)
        
        EPOCHS = 100

#        model.fit(dataset, epochs = EPOCHS, callbacks = [checkpoint_callback])
        model.fit(dataset, epochs = EPOCHS, callbacks = CustomCallback(lowest_loss, checkpoint_prefix))
        
        tf.train.latest_checkpoint(checkpoint_dir)

    def generate_text(model, start_string):    
        num_generate = 210
    
        input_eval = [char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)
        
        #print(input_eval)
    
        text_generated = []

        temperature = .75
    
        model.reset_states()
        for i in range(num_generate):
            if i > 4:
                if text_generated[-4] == ' [N]':
                    break
            
            predictions = model(input_eval)
            predictions = tf.squeeze(predictions, 0)

            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples = 1)[-1,0].numpy()
        
            input_eval = tf.expand_dims([predicted_id], 0)
            
        
            text_generated.append(idx2char[predicted_id])
        
        return (start_string + ''.join(text_generated))
    
    if train == False:
        model = build_model(vocab_size, embedding_dim, rnn_units, batch_size = 1)

        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

        model.build(tf.TensorShape([1, None]))
        
        #model.save('Model/model.hdf5')
        #tf.saved_model.save(model, 'Model')
        
        for x in range(5):
            
            print(generate_text(model, start_string = u"Trump "))

    
#make_data()
AI_stuff(True)
AI_stuff(False)