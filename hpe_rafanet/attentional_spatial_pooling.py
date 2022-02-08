import keras
from keras import backend as K
"""
This code is adapted from the original code in the GitHub repository
https://github.com/CyberZHG/keras-self-attention
"""

class AttentionalPooling(keras.layers.Layer):

    def __init__(self,
                 units=32,
                 return_attention=False,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_additive_bias=True,
                 use_attention_bias=True,
                 **kwargs):
        """Layer initialization.


        :param units: The dimension of the vectors that used to calculate the attention weights.
        :param return_attention: Whether to return the attention weights for visualization.
        :param kernel_initializer: The initializer for weight matrices.
        :param bias_initializer: The initializer for biases.
        :param kernel_regularizer: The regularization for weight matrices.
        :param bias_regularizer: The regularization for biases.
        :param kernel_constraint: The constraint for weight matrices.
        :param bias_constraint: The constraint for biases.
        :param use_additive_bias: Whether to use bias while calculating the relevance of inputs features
                                  in additive mode.
        :param use_attention_bias: Whether to use bias while calculating the weights of attention.
        :param kwargs: Parameters for parent class.
        """
        super(AttentionalPooling, self).__init__(**kwargs)
        self.units = units        
        self.return_attention = return_attention
        self.use_additive_bias = use_additive_bias
        self.use_attention_bias = use_attention_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self._backend = keras.backend.backend()

        self.Wx, self.Wt, self.bh = None, None, None
        self.Wa, self.ba = None, None
        
    def get_config(self):
        config = {
            'units': self.units,
            'return_attention': self.return_attention,
            'use_additive_bias': self.use_additive_bias,
            'use_attention_bias': self.use_attention_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
        }
        base_config = super(AttentionalPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self._build_attention(input_shape)        
        super(AttentionalPooling, self).build(input_shape)

    def _build_attention(self, input_shape):
        feature_dim = int(input_shape[2])

        self.Wt = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wt'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        self.Wx = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wx'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_additive_bias:
            self.bh = self.add_weight(shape=(self.units,),
                                      name='{}_Add_bh'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

        self.Wa = self.add_weight(shape=(self.units, 1),
                                  name='{}_Add_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_attention_bias:
            self.ba = self.add_weight(shape=(1,),
                                      name='{}_Add_ba'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)


    def call(self, inputs, mask=None, **kwargs):

        e = self._call_emission(inputs)

        # Eqn 3: \tau_{n,n'} = \frac{\text{exp}(\sigma_{n,n'})}{\sum_{n'=1}^{N}\text{exp}(\sigma_{n,n'})}
        e = K.exp(e - K.max(e, axis=-1, keepdims=True))
        a = e / K.sum(e, axis=-1, keepdims=True)

        # Eqn 3: a_n = \sum_{n'=1}^{N} \tau_{n, n'} p_{n'}
        v = K.batch_dot(a, inputs)
        
        if self.return_attention:
            return [v, a]
        return v

    def _call_emission(self, inputs):
        input_shape = K.shape(inputs)
        batch_size, input_len = input_shape[0], input_shape[1]

        # Eqn 3: \rho_{n, n'} = \tanh(p_n W_\rho + p_{n'} W_{\rho}' + b_\rho)
        q = K.expand_dims(K.dot(inputs, self.Wt), 2)
        k = K.expand_dims(K.dot(inputs, self.Wx), 1)
        if self.use_additive_bias:
            h = K.tanh(q + k + self.bh)
        else:
            h = K.tanh(q + k)

        # \sigma_{n, n'} = W_\sigma \rho_{n, n'} + b_\sigma
        if self.use_attention_bias:
            e = K.reshape(K.dot(h, self.Wa) + self.ba, (batch_size, input_len, input_len))
        else:
            e = K.reshape(K.dot(h, self.Wa), (batch_size, input_len, input_len))
        return e


    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        if self.return_attention:
            attention_shape = (input_shape[0], output_shape[1], input_shape[1])
            return [output_shape, attention_shape]
        return output_shape


    @staticmethod
    def get_custom_objects():
        return {'AttentionalPooling': AttentionalPooling}

####################################
class WeightedAttention(keras.layers.Layer):

    def __init__(self, use_bias=True, return_attention=False, **kwargs):
        super(WeightedAttention, self).__init__(**kwargs)
        self.use_bias = use_bias
        self.return_attention = return_attention
        self.W, self.b = None, None

    def get_config(self):
        config = {
            'use_bias': self.use_bias,
            'return_attention': self.return_attention,
        }
        base_config = super(WeightedAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.W = self.add_weight(shape=(int(input_shape[2]), 1),
                                 name='{}_W'.format(self.name),
                                 initializer=keras.initializers.get('uniform'))
        if self.use_bias:
            self.b = self.add_weight(shape=(1,),
                                     name='{}_b'.format(self.name),
                                     initializer=keras.initializers.get('zeros'))
        super(WeightedAttention, self).build(input_shape)

    def call(self, x, mask=None):
        #Eqn 4: \psi_n = W_\psi a_n + b_\psi
        logits = K.dot(x, self.W)
        if self.use_bias:
            logits += self.b
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        
        #Eqn 4: \psi_n = W_\psi a_n + b_\psi
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        
        #Eqn 4: x = \sum_{n=1}^{N}a_n W_n 
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return input_shape[0], output_len

    @staticmethod
    def get_custom_objects():
        return {'WeightedAttention': WeightedAttention}
