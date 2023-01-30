import time
import tensorflow as tf
from tensorflow.keras.layers import Dense, MultiHeadAttention, LayerNormalization
from tensorflow.keras import Model

class TimeSpaceEmbedding(tf.keras.layers.Layer):
    def __init__(self, *args):
        super().__init__()

        self.concat = tf.keras.layers.Concatenate(axis = -1)
        self.add = tf.keras.layers.Add()

    def build(self, input_shape):
        window_size = input_shape[0][1]
        num_station = input_shape[1][1]

        self.t_embedding = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.RepeatVector(num_station),
            tf.keras.layers.Reshape(target_shape = (num_station,window_size,8)),
            tf.keras.layers.Permute(dims = (2,1,3))
        ])

        self.s_embedding =  tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.RepeatVector(window_size),
            tf.keras.layers.Reshape(target_shape = (window_size,num_station,2)),
            # tf.keras.layers.LayerNormalization()
        ])

    def call(self, inputs):
        time_embeddings, spatial_embeddings, status = inputs

        time_embeddings = self.t_embedding(time_embeddings)
        spatial_embeddings = self.s_embedding(spatial_embeddings)
        embeddings = self.concat([time_embeddings, spatial_embeddings, status[:,:,:,tf.newaxis]])
        return embeddings
        # return time_embeddings, spatial_embeddings

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, normalizer, d_model = 11):
        super().__init__()
        self.norm = normalizer
        self.d_model = d_model
        self.embedding = tf.keras.layers.Dense(d_model)
        self.ts_embedding = TimeSpaceEmbedding()
        self.add = tf.keras.layers.Add()

    def call(self, inputs):
        x, time_embeddings, spatial_embeddings, status = inputs
        x = self.norm(x) # Shape (batch_size, seq_length)
        x = x[:,:,:,tf.newaxis]
        x = self.embedding(x) #Shape (batch_size, seq_length, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        e = self.ts_embedding([time_embeddings, spatial_embeddings, status])
        x = self.add([x, e])
        return x #Shape (batch_size, seq_length, d_model)

class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True)

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output, attn_scores = self.mha(
            query=x,
            value=x,
            key=x,
            return_attention_scores=True)
        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask = True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
          tf.keras.layers.Dense(dff, activation='relu'),
          tf.keras.layers.Dense(d_model),
          tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x

# class EncoderLayer(tf.keras.layers.Layer):
#     def __init__(self,*, d_model, num_heads, key_dim, dff, dropout_rate=0.1):
#         super().__init__()
#
#         self.self_attention_temporal = GlobalSelfAttention(
#             num_heads=num_heads,
#             key_dim=key_dim,
#             dropout=dropout_rate,
#             attention_axes = (2)
#             )
#
#         # self.self_attention_spatial = GlobalSelfAttention(
#         #     num_heads=num_heads,
#         #     key_dim=d_model,
#         #     dropout=dropout_rate,
#         #     attention_axes = (2))
#
#         self.ffn = FeedForward(d_model, dff)
#
#     def call(self, x):
#         x = self.self_attention_temporal(x)
#         # x = self.self_attention_spatial(x)
#         x = self.ffn(x)
#         return x


# class Encoder(tf.keras.layers.Layer):
#     def __init__(self, *, normalizer, num_layers, d_model, num_heads,
#                key_dim, dff,  dropout_rate=0.1):
#         super().__init__()
#
#         self.d_model = d_model
#         self.num_layers = num_layers
#
#         self.pos_embedding = PositionalEmbedding(
#             normalizer, d_model=d_model)
#
#         self.enc_layers = [
#             EncoderLayer(d_model=d_model,
#                          num_heads=num_heads,
#                          key_dim = key_dim,
#                          dff=dff,
#                          dropout_rate=dropout_rate)
#             for _ in range(num_layers)]
#         self.dropout = tf.keras.layers.Dropout(dropout_rate)
#
#     def call(self, x):
#         # `x` is token-IDs shape: (batch, seq_len)
#
#         # `x`has (tokes, and Temporal and Positional Embeddings)
#         x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.
#
#         # Add dropout.
#         x = self.dropout(x)
#
#         for i in range(self.num_layers):
#             x = self.enc_layers[i](x)
#
#         return x  # Shape `(batch_size, seq_len, d_model)`.

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,
               *,
               d_model,
               num_heads,
               key_dim,
               dff,
               attention_axes,
               dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        # self.causal_self_attention = CausalSelfAttention(
        #     num_heads=num_heads,
        #     key_dim=d_model,
        #     dropout=dropout_rate)

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout_rate,
            attention_axes = attention_axes,
        )

        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
#         x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x
class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, key_dim,
                 dff, attention_axes, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.ts_embeddings = TimeSpaceEmbedding()
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads,
                         key_dim = key_dim, dff=dff,
                         attention_axes = attention_axes,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        self.last_attn_scores = None

    def call(self, time_info, space_info, status, context):

        x = self.ts_embeddings([time_info, space_info, status])

        for i in range(self.num_layers):
            x  = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        # The shape of x is (batch_size, target_seq_len, d_model).
        return x

class Transformer(tf.keras.Model):
    def __init__(self, *, normalizer, num_layers, d_model, num_heads, key_dim,
                    dff, attention_axes, activation, dropout_rate=0.1):
        super().__init__()
        # self.encoder = Encoder(normalizer = normalizer, num_layers=num_layers, d_model=d_model,
        #                        num_heads=num_heads, key_dim = key_dim, dff=dff,
        #                        dropout_rate=dropout_rate)

        self.p = PositionalEmbedding(normalizer = normalizer, d_model = d_model)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, key_dim = key_dim, dff=dff,
                               attention_axes= attention_axes, dropout_rate=dropout_rate)

        self.final_layer = tf.keras.layers.Dense(1, activation=activation)
        self.reshape = tf.keras.layers.Reshape((-1,))
        self.clousures = tf.keras.layers.Multiply()

    def call(self, inputs):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        # context, x = inputs
        context, time_info, space_info, status = inputs

        # context = self.encoder([context, time_info[:,:-1], space_info])  # (batch_size, context_len, d_model)
        context = self.p([context, time_info[:,:-1], space_info, status[:,:-1]])

        x = self.decoder(time_info[:,-1:], space_info, status[:,-1:], context)  # (batch_size, target_len, d_model)

        # Final linear layer output.
        prediction = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del prediction._keras_mask
        except AttributeError:
            pass

        # Return the final output and the attention weights.
        prediction = self.reshape(prediction)
#         prediction = self.clousures([prediction, status])
        return prediction

class MinMax(tf.keras.layers.Layer):
    def __init__(self,*, min_value = None, max_value = None, range_values = None):
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
    def __init__(self,*, mean = None, std = None):
        super().__init__()

        self.mean = mean
        self.std = std
    def adapt(self, data):
        self.mean = tf.math.reduce_mean(data).numpy()
        self.std = tf.math.reduce_std(data).numpy()

    def call(self, x, reverse = False):
        #Raise error is min and max are none.
        if reverse:
            x = (x * self.std) + self.mean
        else:
            x = (x - self.mean)/(self.std)
        return x
