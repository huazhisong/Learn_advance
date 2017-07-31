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
import inception_v4

#%%

N_CLASSES = 100
IMG_W = 299  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 299
BATCH_SIZE = 32
CAPACITY = 200
MAX_STEP = int(1e6) # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 1e-3 # with current parameters, it is suggested to use learning rate<0.0001


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
    train_logits, _ = inception_v4.inception_v4(train_batch, N_CLASSES)
    train_loss = inception_v4.losses(train_logits, train_label_batch)        
    train_op = inception_v4.trainning(train_loss, learning_rate)
    train__acc = inception_v4.evaluation(train_logits, train_label_batch)
       
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
#%%
if __name__ == '__main__':
    run_training()
    
#%%