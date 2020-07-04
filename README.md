# Transfomer-with-TensorFlow2
The repository is an implementation of Transfomer with TensorFlow2 based on the tutorial of official website, which i add some componnents according to the paper.

## Overview
After reading the paper [Attention Is All Your Need](https://arxiv.org/abs/1706.03762), I find the easy implementation in [offcial tutorial](https://tensorflow.google.cn/tutorials/text/transformer), but there are three different details between them. In this repository, I solve two of these details according to the paper.

## Multi-heads
Multi-heads is the first detail. In the paper, the authors use linearly transformation layers to product multi-heads, while the tutorial just split the input into multi-heads easily. In my code, I add a series of dense layers for q, k and v apiece, to product corresponding multi-heads.

## Label Smoothing
Label Smoothing is the second detail. For imporving accuracy and BLEU score of transformer, the authors use the label smoothing as a regularization method, although it would hurts perplexity. 

It should have been a simple question while the loss functions or objections of tensorflow supporting the label smoothing, but I found that when your input's dimension over 2, it doesn't work, even getting much huger loss than without using label smoothing.

And finally, I can only implement the label smoothing function by myself, with smoothing the labels before pass them into the loss functions or objections. Bear in mind, when using this trick, it is no need to set the label smoothing parameter about  functions or objections. By the way, I found someone has submited this bug to tensorflow issue, [see here](https://github.com/tensorflow/tensorflow/issues/39329).

## Sharing Weights
Sharing Weights is the third detail. The author say, they share the same weight matrix between the two embedding layers and the pre-softmax linear transformation. But I couldn't implement this detail in my code for the lacking of time and my personal ability. If you  are interest in this detail, you can continue with my work, or maybe someday I will study the related papers or the author's real source code to solve it.

## About my code
 - tensorflow version: 2.1.0 
 - components.py: basic components of transfomer, like positional encoding and mask.
 - layers.py: multi-heads layer, encode layer, decoder layer and so on.
 - model.py: transformer, loss, optimizer and so on.
 - train.py: entire training preprocess
 - evaluate.py: simple evaluation
 - weights: I only train transformer 3 epochs in a samller architecture, so if you load these weights, it may don't work really well.
