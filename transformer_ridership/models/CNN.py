import tensorflow as tf
from models.shared_layers import ExternalLayer, closure_dummy, closure_mask


class CNNlayer(tf.keras.layers.Layer):
    def __init__(self ):
        super().__init__()
       
        self.dense = tf.keras.layers.Dense(128)
        
    def build(self, input_shape):
        #  
        num_stations = input_shape[-1]
        self.Conv = tf.keras.Sequential([
            tf.keras.layers.Conv1D(num_stations, 3, activation='relu', padding = 'same'),
            tf.keras.layers.Conv1D(num_stations, 3, activation='relu', padding = 'same'),
            tf.keras.layers.Conv1D(num_stations, 3, activation='relu', padding = 'same'),
            tf.keras.layers.Dense(num_stations), 
        ])
        self.built = True

    def call(self, inputs):

        x = self.Conv(inputs)
        x = tf.transpose(x, perm = [0, 2, 1])
        x = self.dense(x)
        return x
    
class CNN(tf.keras.Model):
    def __init__(self, normalizer, closure_mode=None, name='cnn',**kwargs):
        super(CNN, self).__init__( **kwargs)
        
        self.normalizer = normalizer
        self.closure_mode = closure_mode
        self.cnn = CNNlayer()
        self.mask = closure_mask()
        self.dummy = closure_dummy()

        
    def build(self, input_shape): 
        num_station = input_shape[0][-1]
        self.external = ExternalLayer(num_station)
        
        
    def call(self, inputs):
        features, time, status = inputs 
        
        features = self.normalizer(features)
        x1 = self.cnn(features)
        x2 = self.external(time) #Only last 
        
        if self.closure_mode == 'mask':
            x = self.mask([x1,x2], status = status)
        
        elif self.closure_mode == 'dummy':
            x = self.dummy([x1,x2], status = status)
        
        else:
            x = self.mask([x1,x2], status = None)
        
        return x
    
    
# class CNN_dummy(tf.keras.Model):
#     def __init__(self, normalizer, **kwargs):
#         super(DeepPF_dummy, self).__init__( **kwargs)
        
#         self.normalizer = normalizer
#         self.cnn = CNNlayer()
#         self.dummy = closure_dummy()

        
#     def build(self, input_shape): 
#         num_station = input_shape[0][-1]
#         self.external = ExternalLayer(num_station)
        
        
#     def call(self, inputs):
#         features, time, status = inputs 
        
#         features = self.normalizer(features)
#         x1 = self.cnn(featrues)
#         x2 = self.external(time)
#         x = self.dummy([x1,x2], status)
        
#         return x
    
    
# class CNN_mask(tf.keras.Model):
#     def __init__(self, normalizer, **kwargs):
#         super(DeepPF_dummy, self).__init__( **kwargs)
        
#         serlf.normalizer = normalizer
#         self.cnn = CNNlayer()
#         self.mask = closure_mask()

        
#     def build(self, input_shape): 
#         num_station = input_shape[0][-1]
#         self.external = ExternalLayer(num_station)
        
        
#     def call(self, inputs):
#         features, time, status = inputs 
        
#         features = self.normalizer(features)
#         x1 = self.cnn(featrues)
#         x2 = self.external(time)
#         x = self.mask([x1,x2], status)
        
#         return x