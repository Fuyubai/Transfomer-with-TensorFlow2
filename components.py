# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 23:48:52 2020

@author: Morning
"""


import tensorflow as tf
import numpy as np

'''
Positional encoding
'''
def get_angles(pos, i, d_model):
    angle_rates = pos / np.power(10000, (2 * (i//2)) / np.float32(d_model)) 
    return angle_rates

def positional_encoding(pos, d_model):
    angle_rads = get_angles(np.arange(pos)[:, np.newaxis],
                             np.arange(d_model)[np.newaxis, :],
                             d_model)
    
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    # [1, seq_len, dim_embedding]
    # 在axis=0增加维度，以便通过广播和batch匹配
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, tf.float32)

'''
Mask
'''
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    
    # seq: [batch, seq_len_in] => [batch, head, seq_len_out, seq_len_in]
    # 因为每个head的每个输出句子对应的输入句子都是一样的，可以通过广播实现相加
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones([size,size]), -1, 0)
    # mask: [seq_len_out, seq_len_in] while seq_len == seq_out == size
    # 可以通过广播和padding_mask进行运算
    return mask

def create_mask(inp, tar):
    enc_padding_mask = create_padding_mask(inp)    
    dec_padding_mask = create_padding_mask(inp)
    
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_padding_mask_target = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_padding_mask_target, look_ahead_mask)
    
    return enc_padding_mask, combined_mask, dec_padding_mask

'''
Scaled dot product attention 
'''
def scaled_dot_product_attention(q, k, v, mask):
    '''
    Parameters
    ----------
    q : [batch, head, seq_len_q, depth_q]
    k : [batch, head, seq_len_k, depth_k]
    v : [batch, head, seq_len_v, depth_v]
    
    seq_len_q == seq_len_out
    seq_len_k == seq_len_v == seq_len_in
    depth_q == depth_k == depth_v (模型的所有输出都具有相同的维度)
    
    mask : 

    Returns
    -------
    None.
    '''
    
    # [..., seq_len_q, seq_len_k]
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    
    # 缩放
    # [..., seq_len_q, seq_len_k]
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    # [..., seq_len_q, seq_len_k]
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    
    # [..., seq_len_q, depth_v]
    output = tf.matmul(attention_weights, v)
    
    return output, attention_weights

if __name__ == '__main__':
    pass