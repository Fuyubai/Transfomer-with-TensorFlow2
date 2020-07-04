# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 14:46:16 2020

@author: Morning
"""


import tensorflow as tf
from layers import Encoder, Decoder

'''
Transformer
'''
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 input_vocab_size, target_vocab_size, 
                 pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        
        self.Encoder = Encoder(num_layers, d_model, num_heads, dff, 
                               input_vocab_size, pe_input, rate)
        
        self.Decoder = Decoder(num_layers, d_model, num_heads, dff, 
                               target_vocab_size, pe_target, rate)
        
        self.dense = tf.keras.layers.Dense(target_vocab_size)
        
    def call(self, inp, tar, training, 
             enc_padding_mask, look_head_mask, dec_padding_mask):
        
        enc_output = self.Encoder(inp, training, enc_padding_mask)
        
        dec_output, attention_weights = self.Decoder(tar, enc_output, training, look_head_mask, dec_padding_mask)
        
        output = self.dense(dec_output)
        
        return output, attention_weights

'''
Optimizer
'''
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.minimum(arg1, arg2)
    
def optimizer_adam(d_model, warmup_steps=4000, beta_1=0.9, beta_2=0.98, epsilon=1e-9):
    learning_rate = CustomSchedule(d_model, warmup_steps)
    return tf.keras.optimizers.Adam(learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
  
'''
Loss
'''
def loss_function(real, pred, n_classes=None, rate=0.1):    
    if n_classes is None:
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        loss = loss_object(real, pred)
    else:
        # using label smoothing
        # smoothing the label handy and then pass it into loss funciton
        real_one_hot = tf.one_hot(tf.cast(real, tf.int32), n_classes)
        factor = real_one_hot * rate * (1 + 1 / n_classes)
        real_one_hot = real_one_hot + rate / n_classes - factor
        loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')
        loss = loss_object(real_one_hot, pred)

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, loss.dtype)
    loss *= mask
    return tf.reduce_mean(loss)
  
if __name__ == '__main__':
    real = tf.ones([60,60])
    #print(real)
    pred = tf.random.uniform([60,60,8000])
    a = loss_function(real, pred)
    print(a)
    b = loss_function(real, pred, 8000, rate=0.1)
    print(b)
