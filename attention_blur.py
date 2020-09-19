import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects

def _conv_layer(filters, kernel_size, strides=(1, 1), padding='same', name=None):
    return Conv2D(filters, kernel_size, strides=strides, padding=padding,
                  use_bias=True, kernel_initializer='he_normal', name=name)

def _normalize_depth_vars(depth_k, depth_v, filters):

    if type(depth_k) == float:
        depth_k = int(filters * depth_k)
    else:
        depth_k = int(depth_k)

    if type(depth_v) == float:
        depth_v = int(filters * depth_v)
    else:
        depth_v = int(depth_v)

    return depth_k, depth_v

def shape_list(x):
    """Return list of dims, statically where possible."""
    static = x.get_shape().as_list()
    shape = tf.shape(x)
    ret = []
    for i, static_dim in enumerate(static):
        dim = static_dim or shape[i]
        ret.append(dim)
    return ret

class AttentionAugmentation2D(Layer):

    def __init__(self, depth_k, depth_v, num_heads, relative=True, **kwargs):

        super(AttentionAugmentation2D, self).__init__(**kwargs)

        if depth_k % num_heads != 0:
            raise ValueError('`depth_k` (%d) is not divisible by `num_heads` (%d)' % (
                depth_k, num_heads))

        if depth_v % num_heads != 0:
            raise ValueError('`depth_v` (%d) is not divisible by `num_heads` (%d)' % (
                depth_v, num_heads))

        if depth_k // num_heads < 1.:
            raise ValueError('depth_k / num_heads cannot be less than 1 ! '
                             'Given depth_k = %d, num_heads = %d' % (
                             depth_k, num_heads))

        if depth_v // num_heads < 1.:
            raise ValueError('depth_v / num_heads cannot be less than 1 ! '
                             'Given depth_v = %d, num_heads = %d' % (
                                 depth_v, num_heads))

        self.depth_k = depth_k
        self.depth_v = depth_v
        self.num_heads = num_heads
        self.relative = relative

        self.axis = -1

    def build(self, input_shape):
        self._shape = input_shape

        # normalize the format of depth_v and depth_k
        self.depth_k, self.depth_v = _normalize_depth_vars(self.depth_k, self.depth_v,
                                                           input_shape)

        if self.axis == 1:
            _, channels, height, width = input_shape
            
        else:
            _, height, width, channels = input_shape

        if self.relative:
            dk_per_head = self.depth_k // self.num_heads

            if dk_per_head == 0:
                print('dk per head', dk_per_head)
                
            self.key_relative_w = self.add_weight('key_rel_w',
                                                  shape=[2 * width - 1, dk_per_head],
                                                  initializer=initializers.RandomNormal(
                                                      stddev=dk_per_head ** -0.5))
            
            self.key_relative_h = self.add_weight('key_rel_h',
                                                  shape=[2 * height - 1, dk_per_head],
                                                  initializer=initializers.RandomNormal(
                                                      stddev=dk_per_head ** -0.5))

        else:
            self.key_relative_w = None
            self.key_relative_h = None

    def call(self, inputs, **kwargs):
        if self.axis == 1:
            # If channels first, force it to be channels last for these ops
            inputs = tf.transpose(inputs, [0, 2, 3, 1])

        q, k, v = tf.split(inputs, [self.depth_k, self.depth_k, self.depth_v], axis=-1)

        q = self.split_heads_2d(q)
        k = self.split_heads_2d(k)
        v = self.split_heads_2d(v)
        
        # scale query
        depth_k_heads = self.depth_k / self.num_heads
        q *= (depth_k_heads ** -0.5)


        # [Batch, num_heads, height * width, depth_k or depth_v] if axis == -1
        qk_shape = [self._batch, self.num_heads, self._height * self._width, self.depth_k // self.num_heads]
        v_shape = [self._batch, self.num_heads, self._height * self._width, self.depth_v // self.num_heads]
        flat_q = tf.reshape(q, tf.stack(qk_shape))
        flat_k = tf.reshape(k, tf.stack(qk_shape))
        flat_v = tf.reshape(v, tf.stack(v_shape))

        # [Batch, num_heads, HW, HW]
        logits = tf.matmul(flat_q, flat_k, transpose_b=True)


        # Apply relative encodings
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits

        weights = tf.keras.activations.softmax(logits, axis=-1)
        attn_out = tf.matmul(weights, flat_v)

        attn_out_shape = [self._batch, self.num_heads, self._height, self._width, self.depth_v // self.num_heads]

        attn_out_shape = tf.stack(attn_out_shape)
        attn_out = tf.reshape(attn_out, attn_out_shape)
        
        attn_out = self.combine_heads_2d(attn_out)

        # [batch, height, width, depth_v]

        return attn_out

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.axis] = self.depth_v
        return tuple(output_shape)


        
    def split_heads_2d(self, ip):
        batch, height, width, channels = shape_list(ip)
        # Save the spatial tensor dimensions
        self._batch = batch
        self._height = height
        self._width = width
        
        ret_shape = [batch, height, width, self.num_heads, channels // self.num_heads]
        
        split = tf.reshape(ip, ret_shape)
        transpose_axes = (0, 3, 1, 2, 4)
        split = tf.transpose(split, transpose_axes)
        return split

    def relative_logits(self, q):
        shape = tf.shape(q)
        # [batch, num_heads, H, W, depth_v]
        shape = [shape[i] for i in range(5)]

        height = shape[2]
        width = shape[3]

        rel_logits_w = self.relative_logits_1d(q, self.key_relative_w, height, width,
                                               transpose_mask=[0, 1, 2, 4, 3, 5])

        rel_logits_h = self.relative_logits_1d(
            tf.transpose(q, [0, 1, 3, 2, 4]),
            self.key_relative_h, width, height,
            transpose_mask=[0, 1, 4, 2, 5, 3])

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, transpose_mask):
        rel_logits = tf.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = tf.reshape(rel_logits, [-1, self.num_heads * H, W, 2 * W - 1])
        rel_logits = self.rel_to_abs(rel_logits)
        rel_logits = tf.reshape(rel_logits, [-1, self.num_heads, H, W, W])
        rel_logits = tf.expand_dims(rel_logits, axis=3)
        rel_logits = tf.tile(rel_logits, [1, 1, 1, H, 1, 1])
        rel_logits = tf.transpose(rel_logits, transpose_mask)
        rel_logits = tf.reshape(rel_logits, [-1, self.num_heads, H * W, H * W])
        return rel_logits

    def rel_to_abs(self, x):
        shape = tf.shape(x)
        shape = [shape[i] for i in range(3)]
        B, Nh, L, = shape
        col_pad = tf.zeros(tf.stack([B, Nh, L, 1]), dtype=tf.bfloat16)
        x = tf.concat([x, col_pad], axis=3)
        flat_x = tf.reshape(x, [B, Nh, L * 2 * L])
        flat_pad = tf.zeros(tf.stack([B, Nh, L - 1]), dtype=tf.bfloat16)
        flat_x_padded = tf.concat([flat_x, flat_pad], axis=2)
        final_x = tf.reshape(flat_x_padded, [B, Nh, L + 1, 2 * L - 1])
        final_x = final_x[:, :, :L, L - 1:]
        return final_x


    def combine_heads_2d(self, inputs):
        transposed = tf.transpose(inputs, [0, 2, 3, 1, 4])
        self.num_heads, channels = shape_list(transposed)[-2:]
        ret_shape = shape_list(transposed)[:-2]+[self.num_heads*channels]
        return tf.reshape(transposed, ret_shape)
      
      
    def get_config(self):
        config = {
            'depth_k': self.depth_k,
            'depth_v': self.depth_v,
            'num_heads': self.num_heads,
            'relative': self.relative,
        }
        base_config = super(AttentionAugmentation2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def augmented_conv2d(ip, filters, kernel_size=(3, 3), strides=(1, 1),
                     depth_k=0.2, depth_v=0.2, num_heads=8, relative_encodings=True):

    # input_shape = K.int_shape(ip)
    channel_axis = -1

    depth_k, depth_v = _normalize_depth_vars(depth_k, depth_v, filters)

    conv_out = _conv_layer(filters - depth_v, kernel_size, strides)(ip)

    # Augmented Attention Block
    qkv_conv = _conv_layer(2 * depth_k + depth_v, (1, 1), strides)(ip)
    attn_out = AttentionAugmentation2D(depth_k, depth_v, num_heads, relative_encodings)(qkv_conv)
    attn_out = _conv_layer(depth_v, kernel_size=(1, 1))(attn_out)

    output = tf.keras.layers.Concatenate(axis=channel_axis)([conv_out, attn_out])
    return output


class Mish(Activation):

    def __init__(self, activation, dtype=tf.bfloat16, **kwargs):
        super(Mish, self).__init__(activation, dtype=dtype, **kwargs)
        self.__name__ = 'Mish'

def mish(inputs):
    x = inputs * tf.math.tanh(tf.math.softplus(inputs))
    # x = tf.cast(x, tf.bfloat16)
    return x
get_custom_objects().update({'Mish': Mish(mish)})



class BlurPool2D(tf.keras.layers.Layer):

    def __init__(self, filt_size=3, stride=2, **kwargs):
        self.strides = (1,stride,stride,1)
        self.filt_size = filt_size
        self.padding = ( (int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)) ), (int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)) ) )
        if(self.filt_size==1):
            self.a = np.array([1.,])
        elif(self.filt_size==2):
            self.a = np.array([1., 1.])
        elif(self.filt_size==3):
            self.a = np.array([1., 2., 1.])
        elif(self.filt_size==4):    
            self.a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):    
            self.a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):    
            self.a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):    
            self.a = np.array([1., 6., 15., 20., 15., 6., 1.])

        super(BlurPool2D, self).__init__(**kwargs)

    def build(self, input_shape):

        k = self.a
        k = k[:,None]*k[None,:]
        k = k / np.sum(k)
        k = np.tile (k[:,:,None,None], (1,1,input_shape[-1],1))
        self.kernel = tf.keras.backend.constant(k, dtype=tf.bfloat16)

    def compute_output_shape(self, input_shape):
        height = input_shape[1] // self.strides[0]
        width = input_shape[2] // self.strides[1] 
        channels = input_shape[3]
        return (input_shape[0], height, width, channels)
    
    def call(self, x):
        x = tf.keras.backend.spatial_2d_padding(x, padding=self.padding)
        x = tf.nn.depthwise_conv2d(x, self.kernel, strides=self.strides, padding='VALID')
        return x

