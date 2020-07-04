# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 15:09:25 2020

@author: Morning
"""

import tensorflow as tf
import tensorflow_datasets as tfds
from model import Transformer, optimizer_adam, loss_function
from components import create_mask  

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

# tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
#     (en.numpy() for pt, en in train_examples), target_vocab_size=2**13)

# tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
#     (pt.numpy() for pt, en in train_examples), target_vocab_size=2**13)

import pickle
with open('tokenizer_en.pickle', 'rb') as handle:
    tokenizer_en = pickle.load(handle)  
with open('tokenizer_en.pickle', 'rb') as handle:
    tokenizer_pt = pickle.load(handle)

BUFFER_SIZE = 20000
BATCH_SIZE = 64
MAX_LENGTH = 40

def encode(lang1, lang2):
  lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
      lang1.numpy()) + [tokenizer_pt.vocab_size+1]

  lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
      lang2.numpy()) + [tokenizer_en.vocab_size+1]
  
  return lang1, lang2

def filter_max_length(x, y, max_length=MAX_LENGTH):
  return tf.logical_and(tf.size(x) <= max_length,
                        tf.size(y) <= max_length)

def tf_encode(pt, en):
  result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
  result_pt.set_shape([None])
  result_en.set_shape([None])

  return result_pt, result_en

train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)

train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([None],[None]))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


val_dataset = val_examples.map(tf_encode)
val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE, padded_shapes=([None],[None]))

if __name__ == '__main__':
    num_layers = 4
    d_model = 128
    num_heads = 8
    dff = 512
    input_vocab_size = tokenizer_pt.vocab_size + 2
    target_vocab_size = tokenizer_en.vocab_size + 2
    dropout_rate = 0.1

    # load model
    tf.keras.backend.clear_session()
    transfomer = Transformer(num_layers, d_model, num_heads, dff,
                             input_vocab_size, target_vocab_size,
                             pe_input=input_vocab_size,
                             pe_target=target_vocab_size,
                             rate=dropout_rate)
    # optimizer
    optimizer = optimizer_adam(d_model)
    
    # loss and accuracy
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        
    # step
    train_step_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.int64),
                        tf.TensorSpec(shape=(None, None), dtype=tf.int64)]
    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        tar_real = tar[:, 1:]
        tar_inp = tar[:, :-1]
        
        enc_padding_mask, combined_mask, dec_padding_mask = create_mask(inp, tar_inp)
        
        with tf.GradientTape() as tape:
            predictions, _ = transfomer(inp, tar_inp, True, 
                                        enc_padding_mask,
                                        combined_mask,
                                        dec_padding_mask)
            
            loss = loss_function(tar_real, predictions, n_classes=target_vocab_size, rate=0.1)
            
        gradients = tape.gradient(loss, transfomer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transfomer.trainable_variables))
        
        train_loss(loss)
        train_accuracy(tar_real, predictions)
        
    # train
    Epochs = 3
    for epoch in range(Epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)
            if batch % 10 == 0:
                print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))
        
            
    transfomer.save_weights('./weights/v1')
    
    