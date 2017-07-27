# 100中图片
# %%
from sklearn.preprocessing import label_binarize
import tensorflow as tf
import numpy as np
import os

# %%
file = 'train_data.txt'
# change data to real predictions
def get_real_label(data, file = 'train_data.txt', trainable=True):
    '''
    Args:
        data: predi
    Returns:
        list of images and labels
    '''
    image_list = []
    label_list = []
    current_dir = os.path.abspath('../../../data/badu_xijiao/train/')
    with open(file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            infos = line.split(' ')
            if trainable:
                image_path = os.path.abspath(
                        os.path.join(current_dir,'train', infos[0] + '.jpg'))
            else:
                image_path = os.path.abspath(
                        os.path.join(current_dir,'test1', infos[0] + '.jpg'))
            label = infos[1]
            if tf.gfile.Exists(image_path):
                image_list.append(image_path)
                label_list.append(label)
    label_dict = np.array(list(set(label_list)), dtype=np.int32)
    label_dict.sort()
    return np.array([label_dict[int(index)] for index in data])
# loading image paths and labels
def get_files(file = 'train_data.txt', trainable=True):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    image_list = []
    label_list = []
#    for file in os.listdir(file_dir):
#        name = file.split(sep='.')
#        if name[0] == 'cat':
#            cats.append(file_dir + file)
#            label_cats.append(0)
#        else:
#            dogs.append(file_dir + file)
#            label_dogs.append(1)
#    print('There are %d cats\nThere are %d dogs' % (len(cats), len(dogs)))
    current_dir = os.path.abspath('../../../data/badu_xijiao/train/')
    with open(file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            infos = line.split(' ')
            if trainable:
                image_path = os.path.abspath(
                        os.path.join(current_dir,'train', infos[0] + '.jpg'))
            else:
                image_path = os.path.abspath(
                        os.path.join(current_dir,'test1', infos[0] + '.jpg'))
            label = infos[1]
            if tf.gfile.Exists(image_path):
                image_list.append(image_path)
                label_list.append(label)
    label = [np.argmax(label) for
             label in label_binarize(label_list, classes=list(set(label_list)))]
    temp = np.array([image_list, label])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list


# %%

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''

    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    ######################################
    # data argumentation should go to here
    ######################################

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    
    
    # Image processing for training the network. Note the many random
    # distortions applied to the image.
    
    reshaped_image = tf.cast(image, tf.float32)
    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(reshaped_image, [image_W, image_H, 3])
    
    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    # NOTE: since per_image_standardization zeros the mean and makes
    # the stddev unit, this likely has no effect see tensorflow#1458.
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)
  
     # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_standardization(image)
    image.set_shape([image_W, image_H, 3])
    
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)

    # you can also use shuffle_batch
    #    image_batch, label_batch = tf.train.shuffle_batch([image,label],
    #                                                      batch_size=BATCH_SIZE,
    #                                                      num_threads=64,
    #                                                      capacity=CAPACITY,
    #                                                      min_after_dequeue=CAPACITY-1)

    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch

# %% TEST
# To test the generated batches of images
# When training the model, DO comment the following codes




#import matplotlib.pyplot as plt
#
#BATCH_SIZE = 2
#CAPACITY = 256
#IMG_W = 300
#IMG_H = 300
#
#file = 'train_data.txt'
#
#image_list, label_list = get_files(file)
#image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#
#with tf.Session() as sess:
#   i = 0
#   coord = tf.train.Coordinator()
#   threads = tf.train.start_queue_runners(coord=coord)
#   
#   try:
#       while not coord.should_stop() and i<1:
#           
#           img, label = sess.run([image_batch, label_batch])
#           
#           # just test one batch
#           for j in np.arange(BATCH_SIZE):
#               print('label: %d' %label[j])
#               plt.imshow(img[j,:,:,:])
#               plt.show()
#           i+=1
#           
#   except tf.errors.OutOfRangeError:
#       print('done!')
#   finally:
#       coord.request_stop()
#   coord.join(threads)

# %%
