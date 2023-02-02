import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")


"""
Shared Layers:
TensorFlow/keras layers that are used accross multiple models. 
"""
class MinMax(tf.keras.layers.Layer):
    def __init__(self,*, min_value=None, max_value=None, range_values=None):
        super().__init__()

        self.min_x = min_value
        self.max_x = max_value
        self.min_t = 0
        self.max_t = 1

        if range_values:
            self.min_t = range_values[0]
            self.max_t = range_values[1]

    def adapt(self, data):
        self.min_x = tf.math.reduce_min(data).numpy()
        self.max_x = tf.math.reduce_max(data).numpy()

    def call(self, x, reverse = False):
        #Raise error is min and max are none.
        if reverse:
            x = (((x - self.min_t)*(self.max_x - self.min_x))/(self.max_t - self.min_t))
            x = x + self.min_x
        else:
            x = (x - self.min_x)/(self.max_x - self.min_x)
            x = x * (self.max_t - self.min_t) + self.min_t #Range values
        return x

class StandardNormlaization(tf.keras.layers.Layer):
    def __init__(self,*, mean=None, std=None):
        super().__init__()

        self.mean = mean
        self.std = std
    def adapt(self, data):
        self.mean = tf.math.reduce_mean(data).numpy()
        self.std = tf.math.reduce_std(data).numpy()

    def call(self, x, reverse=False):
        #Raise error is min and max are none.
        if reverse:
            x = (x * self.std) + self.mean
        else:
            x = (x - self.mean)/(self.std)
        return x


class ExternalLayer(tf.keras.layers.Layer):
    def __init__(self, num_stations):
        super().__init__()
        
        self.num_stations = num_stations
        
        
    def build(self, input_shape):
        num_temporal_features = input_shape[-1]
        self.repeat = tf.keras.layers.RepeatVector(self.num_stations)
        self.reshape_time = tf.keras.layers.Reshape((num_temporal_features,-1))
        self.built = True
    

    def call(self, inputs):
        x = self.repeat(inputs)
        x = self.reshape_time(x)
        x = tf.transpose(x, perm = [0, 2, 1])
        return x
    
class closure_dummy(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.dense = tf.keras.layers.Dense(128, activation = "relu")
        self.final_layer = tf.keras.layers.Dense(1)
        self.reshape = tf.keras.layers.Reshape((-1,))
        

    def call(self, inputs, status):
#         status = tf.transpose(status, perm = [0, 2, 1])
        x = self.concat(inputs)
        x = self.concat([x,status[:,:,tf.newaxis]])
        x = self.dense(x)
        x = self.final_layer(x)
        x = self.reshape(x)
        return x
    
class closure_mask(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.dense = tf.keras.layers.Dense(128, activation = "relu")
        self.final_layer = tf.keras.layers.Dense(1)
        self.mul = tf.keras.layers.Multiply()
        self.reshape = tf.keras.layers.Reshape((-1,))
        

    def call(self, inputs, status=None):
        
        x = self.concat(inputs)
        x = self.dense(x)
        x = self.final_layer(x)
        if status is not None:
#             status = tf.transpose(status, perm = [0, 2, 1])
            status = status[:,:,tf.newaxis]
            x = self.mul([x, status])
        x = self.reshape(x)
        return x