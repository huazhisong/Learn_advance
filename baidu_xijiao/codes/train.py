# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:11:51 2017

@author: song
"""

#%%
import os
import numpy as np
import tensorflow as tf
import input_helper
import alexnet

#%%

N_CLASSES = 100
IMG_W = 224  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 224
BATCH_SIZE = 16
CAPACITY = 200
MAX_STEP = 10000 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001 # with current parameters, it is suggested to use learning rate<0.0001


#%%
def run_training():
    
    # you need to change the directories to yours.
    file = 'train_data.txt'
    logs_train_dir = './logs/train/'
    
    train, train_label = input_helper.get_files(file)
    
    train_batch, train_label_batch = input_helper.get_batch(train,
                                                          train_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE, 
                                                          CAPACITY)  
    train_logits, _ = alexnet.alexnet_v2(train_batch, N_CLASSES)
    train_loss = alexnet.losses(train_logits, train_label_batch)        
    train_op = alexnet.trainning(train_loss, learning_rate)
    train__acc = alexnet.evaluation(train_logits, train_label_batch)
       
    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()
    
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                    break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])
               
            if step % 50 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
            
            if step % 200 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()
    

#%% Evaluate  image
# when training, comment the following codes.
from datetime import datetime

def evaluate_image():
    '''Test one image against the saved models and parameters
    '''
    
    # you need to change the directories to yours.
    file = 'validation_data.txt'
    train, train_label = input_helper.get_files(file)    
    with tf.Graph().as_default() as g:
        BATCH_SIZE = 16
        N_CLASSES = 100
        MAX_STEP = 100
        train_batch, train_label_batch = input_helper.get_batch(train,
                                                          train_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE, 
                                                          CAPACITY)  
        logit, _ = alexnet.alexnet_v2(train_batch, N_CLASSES)
        
        top_k_op = tf.nn.in_top_k(logit, train_label_batch, 1)
        
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
      
        summary_writer = tf.summary.FileWriter('./log/eval', g)
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





