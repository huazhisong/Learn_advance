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
import csv
import os
# import pdb

import alexnet
import input_helper

#%%

BATCH_SIZE = 1
N_CLASSES = 100
MAX_STEP = 10593
# MAX_STEP = 10
IMG_W = 299  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 299
CAPACITY = 200

# %%
def get_files():
    '''
    Returns:
        list of images and image_ids
    '''

    image_dir = os.path.abspath('../../../data/badu_xijiao/test/image/')
    image_files= os.listdir(image_dir)
    image_ids = [file.split('.jpg')[0] for file in image_files]
    image_list = [os.path.join(image_dir, image) for image in image_files]
    return image_list, image_ids
# %%

def get_batch(image, image_id, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        image_id: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        image_id_batch: 1D tensor [batch_size], dtype=tf.int32
    '''

    image = tf.cast(image, tf.string)
    image_id = tf.cast(image_id, tf.string)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, image_id])

    image_id = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    ######################################
    # data argumentation should go to here
    ######################################

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)

    # if you want to test the generated batches of images, you might want to comment the following line.
    image = tf.image.per_image_standardization(image)

    image_batch, image_id_batch = tf.train.batch([image, image_id],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)

    # you can also use shuffle_batch
    #    image_batch, image_id_batch = tf.train.shuffle_batch([image,image_id],
    #                                                      batch_size=BATCH_SIZE,
    #                                                      num_threads=64,
    #                                                      capacity=CAPACITY,
    #                                                      min_after_dequeue=CAPACITY-1)

    image_id_batch = tf.reshape(image_id_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, image_id_batch
# %%

def test_image():
    '''Test images against the saved models and parameters
    '''

    train, image_ids = get_files()    
    with tf.Graph().as_default():
        train_batch, imageid_batch = get_batch(train, image_ids, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)  
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
                    print(batch_predictions)
                    all_predictions = np.concatenate([all_predictions, batch_predictions[0]])

            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                coord.request_stop()    
            coord.join(threads)
    # pdb.set_trace()
    all_predictions.astype(np.int64)
    all_predictions = input_helper.get_real_label(all_predictions)
    predictions_human_readable = np.column_stack((all_predictions, np.array(image_ids)))
    out_path = os.path.join(".", "prediction.csv")
    print("Saving evaluation to {0}".format(out_path))
    with open(out_path, 'w') as f:
	    for line in predictions_human_readable:
		    f.write(str(line[0])+'\t'+str(line[1])+'\n')
#%%
if __name__ == '__main__':
    test_image()

#%%