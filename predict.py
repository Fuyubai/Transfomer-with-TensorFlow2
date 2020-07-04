
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 15:10:33 2020

@author: Morning
"""
import tensorflow as tf
from components import create_mask  
from model import Transformer
import pickle

def evaluate(inp_sentence, transformer, tokenizer_pt, tokenizer_en, MAX_LENGTH):
    start_token = [tokenizer_pt.vocab_size]
    end_token = [tokenizer_pt.vocab_size+1]
    
    # encoder input
    inp_sentence = start_token + tokenizer_pt.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)
    
    # decoder input
    decoder_input = [tokenizer_en.vocab_size]
    output = tf.expand_dims(decoder_input, 0)
    
    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_mask(encoder_input, output)
        
        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)
        # take the last prediction and decode it to the word
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), dtype=tf.int32)
        # if the last word is not the stop token, 
        # concat it with the decoder input and predict again
        if predicted_id == tokenizer_en.vocab_size + 1:
            return tf.squeeze(output, axis=0), attention_weights        
        output = tf.concat([output, predicted_id], axis=-1)        
        return tf.squeeze(output, axis=0), attention_weights
       
if __name__ == '__main__':
    # load tokenizer
    with open('tokenizer_en.pickle', 'rb') as handle:
        tokenizer_en = pickle.load(handle)  
    with open('tokenizer_en.pickle', 'rb') as handle:
        tokenizer_pt = pickle.load(handle)
        
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
    transfomer.load_weights('./weights/v1')
    
    # evaluate
    MAX_LENGTH = 40
    sentence = 'este é um problema que temos que resolver.'
    result, attention_weights = evaluate(sentence,
                                         transfomer,
                                         tokenizer_pt,
                                         tokenizer_en,
                                         MAX_LENGTH)    
    predicted_sentence = tokenizer_en.decode([i for i in result if i < tokenizer_en.vocab_size])
        
    
