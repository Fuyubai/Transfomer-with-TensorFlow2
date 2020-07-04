# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 01:57:45 2020

@author: Morning
"""


import tensorflow as tf
from components import positional_encoding, scaled_dot_product_attention

'''
Multi-head attention
在输入的最后一个维度, 即embedding的维度进行拆分，对各子块并行计算注意力
'''
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        self.num_heads = num_heads
        self.d_model = d_model
        assert self.d_model % self.num_heads == 0        
        self.depth = self.d_model // self.num_heads
        
        self.qw = tf.keras.layers.Dense(self.d_model)
        self.kw = tf.keras.layers.Dense(self.d_model)
        self.vw = tf.keras.layers.Dense(self.d_model)
        
        self.q_split = [tf.keras.layers.Dense(self.depth) for _ in range(num_heads)]
        self.k_split = [tf.keras.layers.Dense(self.depth) for _ in range(num_heads)]
        self.v_split = [tf.keras.layers.Dense(self.depth) for _ in range(num_heads)]
        
        self.connect = tf.keras.layers.Dense(self.d_model)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, [batch_size, -1, self.num_heads, self.depth])
        return tf.transpose(x, perm=[0,2,1,3])
    
    def call(self, v, k, q, mask): # 注意入参的顺序是v, k, q
        batch_size = tf.shape(q)[0]
        
        q = self.qw(q)  # [batch_size, seq_len_q, d_model]
        k = self.kw(k)  # [batch_size, seq_len_k, d_model]
        v = self.vw(v)  # [batch_size, seq_len_v, d_model]
        
        # [[batch_size, seq_len_q, depth] * num_heads]
        q_heads = [self.q_split[i](q) for i in range(self.num_heads)]
        k_heads = [self.k_split[i](k) for i in range(self.num_heads)]
        v_heads = [self.v_split[i](v) for i in range(self.num_heads)]
        
        # [batch_size, num_heads, seq_len_q, depth]
        q = tf.stack(q_heads, axis=1)
        k = tf.stack(k_heads, axis=1)
        v = tf.stack(v_heads, axis=1)
        
        # q = self.split_heads(q, batch_size)  # [batch_size, num_heads, seq_len_q, depth]
        # k = self.split_heads(k, batch_size)  # [batch_size, num_heads, seq_len_k, depth]
        # v = self.split_heads(v, batch_size)  # [batch_size, num_heads, seq_len_v, depth]
        
        # scaled_dot_product: [batch_size, num_heads, seq_len_q, depth]
        # attention_weights: [batch_size, num_heads, seq_len_q, seq_len_v]
        scaled_dot_product, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        # scaled_dot_product: [batch_size, seq_len_q, num_heads, depth]
        scaled_dot_product = tf.transpose(scaled_dot_product, perm=[0,2,1,3])
        # scaled_dot_product: [batch_size, seq_len_q, d_model]
        scaled_dot_product = tf.reshape(scaled_dot_product, [batch_size, -1, self.d_model])    
        
        output = self.connect(scaled_dot_product)
        
        return output, attention_weights
    
'''
Point wise feed forword netword
'''
def point_wise_feed_forword_netword(d_model, dff):
    return tf.keras.Sequential([tf.keras.layers.Dense(dff, activation='relu'),
                                tf.keras.layers.Dense(d_model)])

'''
Encoder layer
'''
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forword_netword(d_model, dff)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        
    def call(self, x, training, mask):
        mha_out, _ = self.mha(x, x, x, mask) # [batch_size, seq_len, d_model]
        mha_out = self.dropout1(mha_out, training=training)
        mha_out = self.layernorm1(x + mha_out) # [batch_size, seq_len, d_model]
        
        ffn_out = self.ffn(mha_out) # [batch_size, seq_len, d_model]
        ffn_out = self.dropout2(ffn_out, training=training)
        ffn_out = self.layernorm2(mha_out + ffn_out) # [batch_size, seq_len, d_model]
        
        return ffn_out
        
'''
Decoder layer
'''
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forword_netword(d_model, dff)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        mha1_out, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        mha1_out = self.dropout1(mha1_out, training=training)
        mha1_out = self.layernorm1(x + mha1_out)
        
        # enc_output: [batch_size, seq_len_in, d_model] => v, k
        # x: [batch_size, seq_len_out, d_model] => q
        # attn_weights_block2: [batch_size, num_heads, seq_len_out, seq_len_in]
        mha2_out, attn_weights_block2 = self.mha2(enc_output, enc_output, mha1_out, padding_mask)
        mha2_out = self.dropout2(mha2_out, training=training)
        mha2_out = self.layernorm2(mha1_out + mha2_out)
        
        ffn_out = self.ffn(mha2_out)
        ffn_out = self.dropout3(ffn_out, training=training)
        ffn_out = self.layernorm3(mha2_out + ffn_out)

        return ffn_out, attn_weights_block1, attn_weights_block2
        
'''
Encoder
'''
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        
        self.dropout = tf.keras.layers.Dropout(rate)
        
    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        
        # [batch_size, seq_len]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        # [batch_size, seq_len, d_model]
        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
            
        # [batch_size, seq_len, d_model]    
        return x
            
'''
Decoder
'''
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()
                
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        
        self.dropout = tf.keras.layers.Dropout(rate)
        
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attention_weights = {}
        seq_len = tf.shape(x)[1]
        
        # [batch_size, seq_len_out]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        # [batch_size, seq_len_out, d_model]
        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            # block1, block2: [batch_size, num_heads, seq_len_out, seq_len_in]
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
        
        # [batch_size, seq_len_out, d_model]    
        return x, attention_weights
    
if __name__ == '__main__':
    sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8, 
                             dff=2048, input_vocab_size=8500,
                             maximum_position_encoding=10000)
    
    sample_encoder_output = sample_encoder(tf.random.uniform((64, 62)), 
                                           training=False, mask=None)
    
    print (sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)
    
    
    sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8, 
                         dff=2048, target_vocab_size=8000,
                         maximum_position_encoding=5000)

    output, attn = sample_decoder(tf.random.uniform((64, 26)), 
                                  enc_output=sample_encoder_output, 
                                  training=False, look_ahead_mask=None, 
                                  padding_mask=None)
    
    print(output.shape, attn['decoder_layer2_block2'].shape)