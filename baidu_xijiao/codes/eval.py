# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 19:42:36 2017

@author: song
"""

#%% Evaluate  image
# when training, comment the following codes.
from datetime import datetime
import tensorflow as tf
import numpy as np

import input_helper
import alexnet

#%%

BATCH_SIZE = 16
N_CLASSES = 100
MAX_STEP = 10000
IMG_W = 299  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 299
CAPACITY = 200

def evaluate_image():
    '''Test one image against the saved models and parameters
    '''
    
    # you need to change the directories to yours.
    file = 'validation_data.txt'
    train, train_label = input_helper.get_files(file)    
    with tf.Graph().as_default() as g:
        train_batch, train_label_batch = input_helper.get_dev_batch(train,
                                                                    train_label,
                                                                    IMG_W,
                                                                    IMG_H,
                                                                    BATCH_SIZE, 
                                                                    CAPACITY)  
        logit, _ = alexnet.alexnet_v2(train_batch, N_CLASSES)
        
        top_k_op = tf.nn.in_top_k(logit, train_label_batch, 1)
        
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
      
        summary_writer = tf.summary.FileWriter('./logs/eval', g)
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
            total_sample_count = MAX_STEP * BATCH_SIZE
            true_count = 0  # Counts the number of correct predictions.
            try:
                for step in np.arange(MAX_STEP):
                    if coord.should_stop():
                        break
                    predictions  = sess.run([top_k_op])
                    true_count += np.sum(predictions)
                
                # Compute precision @ 1.
                precision = true_count / total_sample_count
                print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
                
                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op))
                summary.value.add(tag='Precision @ 1', simple_value=precision)
                summary_writer.add_summary(summary, global_step)
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                coord.request_stop()    
            coord.join(threads)

#%%
if __name__ == '__main__':
    evaluate_image()
    
#%%