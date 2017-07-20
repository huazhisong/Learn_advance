# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 19:46:08 2017

@author: song
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 19:42:36 2017

@author: song
"""

#%% Evaluate  image
# when training, comment the following codes.
import tensorflow as tf
import numpy as np
import os
import csv

import alexnet

#%%

BATCH_SIZE = 16
N_CLASSES = 100
MAX_STEP = 100
IMG_W = 224  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 224
CAPACITY = 200

def get_files():
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''

    image_dir = os.path.abspath('../../../data/badu_xijiao/test/image/')
    image_files= os.listdir(image_dir)
    image_list = [os.path.join(image_dir, image) for image in image_files]
    return image_list

def get_batch(image, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''

    image = tf.cast(image, tf.string)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image])

    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    ######################################
    # data argumentation should go to here
    ######################################

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)

    # if you want to test the generated batches of images, you might want to comment the following line.
    image = tf.image.per_image_standardization(image)

    image_batch = tf.train.batch([image],
                                 batch_size=batch_size,
                                 num_threads=64,
                                 capacity=capacity)
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch

def test_image():
    '''Test images against the saved models and parameters
    '''

    train = get_files()    
    with tf.Graph().as_default():
        train_batch, train_label_batch = get_batch(train, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)  
        logit, _ = alexnet.alexnet_v2(train_batch, N_CLASSES)
        
        prediction = tf.argmax(logit, 1)
        
        # you need to change the directories to yours.
        logs_train_dir = './logs/train/' 
                       
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            all_predictions = []
            try:
                for step in np.arange(MAX_STEP):
                    if coord.should_stop():
                        break
                    batch_predictions  = sess.run([prediction])
                    all_predictions = np.concatenate([all_predictions, batch_predictions])
                    print(all_predictions.shape)

            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                coord.request_stop()    
            coord.join(threads)
            
#    predictions_human_readable = np.column_stack((np.array(train), all_predictions))
#    out_path = os.path.join(".", "prediction.csv")
#    print("Saving evaluation to {0}".format(out_path))
#    with open(out_path, 'w') as f:
#        csv.writer(f).writerows(predictions_human_readable)       

#%%
test_image()