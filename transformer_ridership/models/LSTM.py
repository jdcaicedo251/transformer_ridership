import tensorflow as tf
from models.shared_layers import ExternalLayer, closure_dummy, closure_mask

"""
Reference:
Liu, Y., Liu, Z., & Jia, R. (2019). DeepPF: A deep learning based architecture for metro passenger 
flow prediction. Transportation Research Part C: Emerging Technologies, 101, 18–34. https://doi.org/https://doi.org/10.1016/j.trc.2019.01.027
"""


class LSTMLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()

        
    def build(self, input_shape):
        # Builds LSTM units for each transit stations. 
        num_stations = input_shape[-1]
        self.lstm_1 = tf.keras.layers.LSTM(num_stations, return_sequences=True)
        self.lstm_2 = tf.keras.layers.LSTM(num_stations, return_sequences=True)
        self.lstm_3 = tf.keras.layers.LSTM(num_stations, return_sequences=True)
        self.built = True

        
    def call(self, inputs):
        x = self.lstm_1(inputs)
        x = self.lstm_2(x)
        x = self.lstm_3(x)
        x = tf.transpose(x, perm = [0, 2, 1])
        return x
    
    
class DeepPF(tf.keras.Model):
    def __init__(self, normalizer, closure_mode=None, name='lstm', **kwargs):
        super(DeepPF, self).__init__( **kwargs)
        
        self.normalizer = normalizer
        self.closure_mode = closure_mode
        self.sameday = LSTMLayer()
        self.day = LSTMLayer()
        self.week = LSTMLayer()
        self.mask = closure_mask()
        self.dummy = closure_dummy()

        
    def build(self, input_shape): 
        num_station = input_shape[0][-1]
        self.external = ExternalLayer(num_station)
        
        
    def call(self, inputs):
        features, time, status = inputs 
        
        features = self.normalizer(features)
        sameday = features[:,-6:,:]
        day = features[:,2:4,:]
        week = features[:,:2,:]
        
        x1 = self.sameday(sameday)
        x2 = self.day(day)
        x3 = self.week(week)
        x4 = self.external(time)
    
        if self.closure_mode == 'mask':
            x = self.mask([x1,x2,x3,x4],status)
        
        elif self.closure_mode == 'dummy':
            x = self.dummy([x1,x2,x3,x4],status)
        
        else:
            x = self.mask([x1,x2,x3,x4],status = None)
        
        return x