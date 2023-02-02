import tensorflow as tf
from models.shared_layers import ExternalLayer, closure_dummy, closure_mask

class FNNlayer(tf.keras.layers.Layer):
    def __init__(self ):
        super().__init__()
        
        self.temporalDense = tf.keras.Sequential([
            tf.keras.layers.Dense(units = 256),
            tf.keras.layers.Dense(units = 256),
            tf.keras.layers.Dense(units = 128),
            tf.keras.layers.Dense(units = 64)
        ])
        
        
    def build(self, input_shape):
        #  
        num_stations = input_shape[-1]
        self.spatialDense = tf.keras.Sequential([
            tf.keras.layers.Dense(units = 735),
            tf.keras.layers.Dense(units = 735),
            tf.keras.layers.Dense(units = 735),
            tf.keras.layers.Dense(units = num_stations)
        ])
        self.built = True

        
    def call(self, inputs):
        x = self.spatialDense(inputs)
        x = tf.transpose(x, perm = [0, 2, 1])
        x = self.temporalDense(x)
        return x
    
    
class FNN(tf.keras.Model):
    def __init__(self, normalizer, closure_mode=None, name='fnn',**kwargs):
        super(FNN, self).__init__( **kwargs)
        
        self.normalizer = normalizer
        self.closure_mode = closure_mode
        self.dense = FNNlayer()
        self.mask = closure_mask()
        self.dummy = closure_dummy()

        
    def build(self, input_shape): 
        num_station = input_shape[0][-1]
        self.external = ExternalLayer(num_station)
        
        
    def call(self, inputs):
        features, time, status = inputs 
        
        features = self.normalizer(features) 
        x1 = self.dense(features)
        x2 = self.external(time)
        
        if self.closure_mode == 'mask':
            x = self.mask([x1,x2],status)
        
        elif self.closure_mode == 'dummy':
            x = self.dummy([x1,x2], status)
        
        else:
            x = self.mask([x1,x2], status = None)
        
        return x
    
    
# class FNN_dummy(tf.keras.Model):
#     def __init__(self, normlaizer, **kwargs):
#         super(DeepPF_dummy, self).__init__( **kwargs)
        
#         self.normalizer = normalizer
#         self.dense = FNNlayer()
#         self.dummy = closure_dummy()

        
#     def build(self, input_shape): 
#         num_station = input_shape[0][-1]
#         self.external = ExternalLayer(num_station)
        
        
#     def call(self, inputs):
#         features, time, status = inputs 
        
#         features = self.normalizer(features) 
#         x1 = self.dense(featrues)
#         x2 = self.external(time)
#         x = self.dummy([x1,x2], status)
        
#         return x
    
    
# class FNN_mask(tf.keras.Model):
#     def __init__(self, normalizer, **kwargs):
#         super(DeepPF_dummy, self).__init__( **kwargs)
        
#         self.normalizer = normalizer
#         self.dense = FNNlayer()
#         self.mask = closure_mask()

        
#     def build(self, input_shape): 
#         num_station = input_shape[0][-1]
#         self.external = ExternalLayer(num_station)
        
        
#     def call(self, inputs):
#         features, time, status = inputs 
        
#         features = self.normalizer(features) 
#         x1 = self.dense(featrues)
#         x2 = self.external(time)
#         x = self.mask([x1,x2], status)
        
#         return x