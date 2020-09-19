import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from clr import *
from attention_blur import *


#Initialization of the TPU
tpu='...'
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu)
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
tpu_strategy = tf.distribute.experimental.TPUStrategy(resolver)

BATCH_SIZE=256
step_factor=8
EPOCHS=250

# Your model's name
name='......'

# Your TFRecord directory
bucket = '.../*'

tfr = tf.io.gfile.glob(bucket)
random.seed(4)
random.shuffle(tfr)
train = tfr
valid = tfr[int((len(tfr)*0.8)):]
training_filenames = train
validation_filenames = valid
steps_per_epoch = int(len(train)*"(the sum of images per tfrecord)"/(BATCH_SIZE))
val_steps = int(len(valid)*"(the sum of images per tfrecord)"//(BATCH_SIZE))
step_size = steps_per_epoch*step_factor

AUTO = tf.data.experimental.AUTOTUNE


# Functions to load dataset from TFRecords

def read_tfrecord(example):
    """
    Decoding TFRecord to images and labels(normalized to image's size)
    """
    features = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'labels': tf.io.FixedLenFeature([], tf.string)
    }
    sample=tf.io.parse_single_example(example, features)
    image = tf.io.decode_raw(sample['image'], tf.uint8)
    image = tf.reshape(image, [224, 224, 3])
    labels = tf.io.decode_raw(sample['labels'], tf.float64)
    labels = tf.reshape(labels, [21,2])
    labels = tf.dtypes.cast(labels, tf.float32)
    image, lbl = augmentation(image, labels)
    return image, lbl/224.

def augmentation(image, labels):
    """
    Augment dataset with random brightness, saturation, contrast, and image quality. Then it is casted to bfloat16 and normalized
    """
    image = tf.cast(image, tf.uint8)
    image = tf.image.random_jpeg_quality(image,min_jpeg_quality=70,max_jpeg_quality=100)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [224, 224, 3])
    image = tf.image.random_brightness(image, max_delta=25/255)
    image = tf.image.random_saturation(image, lower=0.3, upper=1.7)
    image = tf.image.random_contrast(image, lower=0.3, upper=1.7)
    image = tf.cast(image, tf.bfloat16)
    image = tf.image.per_image_standardization(image)
    return image, labels

def load_dataset(filenames):
    """
    Load each TFRecord
    """
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    files = tf.data.Dataset.list_files(filenames)
    dataset = files.with_options(ignore_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=512, num_parallel_calls=AUTO)
    dataset = dataset.map(map_func=read_tfrecord, num_parallel_calls=AUTO)
    return dataset

def get_batched_dataset(filenames):
    """
    Feeds batch to the fit function
    """
    dataset = load_dataset(filenames)
    dataset = dataset.shuffle(2000)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(AUTO)
    return dataset

# The same as above but without augmentation to the validation dataset

def read_tfrecord1(example):
    features = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'labels': tf.io.FixedLenFeature([], tf.string)
    }
    sample=tf.io.parse_single_example(example, features)
    image = tf.io.decode_raw(sample['image'], tf.uint8)
    image = tf.reshape(image, [224, 224, 3])
    image = augmentation1(image)
    labels = tf.io.decode_raw(sample['labels'], tf.float64)
    labels = tf.reshape(labels, [21,2])
    labels = tf.dtypes.cast(labels, tf.float32)
    return image, labels/224.

def augmentation1(image):
    image = tf.cast(image, tf.bfloat16)
    image = tf.image.per_image_standardization(image)
    return image

def load_dataset1(filenames):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    files = tf.data.Dataset.list_files(filenames)
    dataset = files.with_options(ignore_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=512, num_parallel_calls=AUTO)
    dataset = dataset.map(map_func=read_tfrecord1, num_parallel_calls=AUTO)
    return dataset

def get_batched_dataset1(filenames):
    dataset = load_dataset1(filenames)
    dataset = dataset.shuffle(2000)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(AUTO)
    return dataset


def get_training_dataset():
    return get_batched_dataset(training_filenames)

def get_validation_dataset():
    return get_batched_dataset1(validation_filenames)



# Model's architecture

def avg_pool(inp, pool=2, stride=2):
    x = tf.keras.layers.AveragePooling2D(pool_size=pool, strides=stride)(inp)
    return x

def conv(inp, kernel, filt, dilation=2, stride=1, pad='same'):
    x = layers.Conv2D(filters=filt, kernel_size=kernel,strides=stride, padding=pad, kernel_regularizer=keras.regularizers.l2(0.01))(inp)
    return x

def aug_block(inp, fout, dk, dv, nh, kernel=11):
    x = augmented_conv2d(inp, filters=fout, kernel_size=(kernel, kernel), depth_k=dk, depth_v=dv, num_heads=nh, relative_encodings=True)
    x = layers.BatchNormalization(axis=-1, fused=True)(x)
    x = layers.Activation('Mish')(x)
    return x

def ARB(inp, fout, dk, dv, nh, kernel,  aug=True):
    x = conv(inp, kernel=1, filt=fout*4, pad='same')
    x = layers.BatchNormalization(axis=-1, fused=True)(x)
    x = layers.Activation('Mish')(x)
    x = layers.DepthwiseConv2D(kernel_size=kernel, strides=1, padding='same')(x)
    x = layers.BatchNormalization(axis=-1, fused=True)(x)
    x = layers.Activation('Mish')(x)
    if aug==True:
        a = aug_block(x, fout*4, dk, dv, nh, kernel)
        x = layers.Add()([a, x])
    x = conv(x, kernel=1, filt=fout, pad='same')
    x = layers.BatchNormalization(axis=-1, fused=True)(x)
    x = layers.Activation('Mish')(x)
    return x

def transition(inp, filters):
    x = conv(inp, kernel=1, filt=filters, pad='same')
    x = BlurPool2D()(x)
    x = layers.BatchNormalization(axis=-1, fused=True)(x)
    return x

def dense(x, kernel,num, nh=4, filters=10, aug=True):
    x_list=[x]
    for i in range(num):
        x = ARB(x, filters, 0.1, 0.1, nh,kernel, aug)
        x_list.append(x)
        x = tf.concat(x_list, axis=-1)
    return x

def create_model():
    Input = layers.Input(shape=(224, 224, 3), dtype='bfloat16') #224
    x = dense(Input, 5, num=8, aug=False)
    ###############################
    y = transition(x, 64) #112
    x = dense(y, 5, num=8, aug=False)
    ###############################
    y = transition(x, 64) #56
    x = dense(y, nh=1, kernel=3, num=6)
    ###############################
    y = transition(x, 64) #28
    x = dense(y, nh=4, kernel=3, num=8)
    ###############################
    y = transition(x, 64) #14
    x = dense(y, nh=4, kernel=3, num=10)
    ###############################
    y = transition(x, 64) #7
    x = dense(y, nh=4, kernel=3, num=12)
    ###############################
    y = transition(x, 128) #4
    x = dense(y, nh=4, kernel=3, num=14)
    ###############################
    x = transition(x, 128) #2
    x = dense(x, nh=4, kernel=2, num=32)
    x = aug_block(x, 100, 0.1, 0.1, 10, 2)
    ###############################
    x = avg_pool(x) #1
    x = conv(x, 1, 42, 1, 1)
    x = tf.keras.layers.Lambda(lambda x : tf.keras.activations.relu(x, max_value=1.))(x)
    x = tf.keras.layers.Reshape((21,2))(x)
    x = tf.cast(x, tf.float32)
    model = tf.keras.Model(Input, x)
    model.compile(optimizer=keras.optimizers.SGD(), loss=rmse)
    return model

def rmse(x, y):
    x = tf.math.sqrt(tf.keras.losses.MSE(x,y))
    return x

########### print Learning Rate ###########
lr_print=showLR()

############ Cyclical Learning Rate ###############
clr_triangular = CyclicLR(base_lr= 0.0001, max_lr=0.1, step_size=step_size, mode='triangular2')

######## Tensorboard ############
# Your log directory
logdir='....'+name
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True)

######## Model Save #############
# Your saving directory
filepath = '..../'+name+'.hdf5'
checkpoint = keras.callbacks.ModelCheckpoint((filepath),monitor='val_loss', save_best_only=True, verbose=1, save_weights_only=True)

#################################
clbk=[tensorboard_callback, checkpoint, lr_print, clr_triangular]

###############Fit#############
policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
tf.keras.mixed_precision.experimental.set_policy(policy)

with tpu_strategy.scope():
    model=create_model()
    # print(model.summary())
    model.load_weights(filepath)

model.fit(get_training_dataset(), validation_data=get_validation_dataset(),  initial_epoch=0, steps_per_epoch=steps_per_epoch ,validation_steps=val_steps, epochs=EPOCHS, verbose=1, callbacks=clbk)

