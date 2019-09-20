# Real-time Hair Segmentation and Recoloring on Mobile GPUs (https://arxiv.org/abs/1907.06740)
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, PReLU, MaxPooling2D, Add, Concatenate
## MaxUnpooling2D --> Input: tensor and Argmaxed Tensor --> Ref: https://stackoverflow.com/questions/36548736/tensorflow-unpooling
## MaxPoolingWithArgmax2D --> Compatible with tf.nn.max_pool_with_argmax
## 
TODO