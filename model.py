'''Define the hybrid model base on 10.1109/TCBB.2022.3201631.
'''


__author__ = 'Chao Wu'


import numpy as np
import tensorflow as tf


class Atten(tf.keras.layers.Layer):

    def __init__(self, n_heads, model_dim, dropout_rate):

        super().__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads = n_heads,
            key_dim = model_dim,
            dropout = dropout_rate
        )
        self.norm = tf.keras.layers.LayerNormalization()

    
    def call(self, x, context, return_weight = False):

        mha_output = self.mha(
            query = x,
            value = context,
            key = context,
            return_attention_scores = return_weight
        )
        if return_weight:
            mha_output, atten_weight = mha_output
            output = self.norm(x + mha_output)

            return output, atten_weight
        
        else:
            output = self.norm(x + mha_output)

            return output
    

class FeedForward(tf.keras.layers.Layer):

    def __init__(self, ff_dim, model_dim, dropout_rate):
        
        super().__init__()

        self.dense1 = tf.keras.layers.Dense(ff_dim, activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(model_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.norm = tf.keras.layers.LayerNormalization()


    def call(self, x):

        ff_output = self.dense1(x)
        ff_output = self.dense2(ff_output)
        ff_output = self.dropout(ff_output)
        
        output = self.norm(x + ff_output)
        
        return output


class EncoderBlock(tf.keras.layers.Layer):

    def __init__(self, n_heads, model_dim, ff_dim, dropout_rate):

        super().__init__()

        self.self_attention = Atten(
            n_heads = n_heads,
            model_dim = model_dim,
            dropout_rate = dropout_rate
        )
        self.feedforward = FeedForward(
            ff_dim = ff_dim,
            model_dim = model_dim,
            dropout_rate = dropout_rate
        )


    def call(self, x):

        x = self.self_attention(x, x)
        x = self.feedforward(x)

        return x


class Encoder(tf.keras.layers.Layer):

    def __init__(self, n_blocks, n_heads, model_dim, ff_dim, dropout_rate):

        super().__init__()

        self.n_blocks = n_blocks        
        self.encoder_blocks = [
            EncoderBlock(
                n_heads = n_heads, 
                model_dim = model_dim, 
                ff_dim = ff_dim, 
                dropout_rate = dropout_rate)
            for _ in range(n_blocks)
        ]


    def call(self, x):

        for i in range(self.n_blocks):
            x = self.encoder_blocks[i](x)

        return x


class DecoderBlock(tf.keras.layers.Layer):

    def __init__(self, n_heads, model_dim, ff_dim, dropout_rate):

        super().__init__()

        self.self_attention = Atten(   
            n_heads = n_heads,
            model_dim = model_dim,
            dropout_rate = dropout_rate
        )
        self.cross_attention = Atten(
            n_heads = n_heads,
            model_dim = model_dim,
            dropout_rate = dropout_rate
        )
        self.feedforward = FeedForward(
            ff_dim = ff_dim,
            model_dim = model_dim,
            dropout_rate = dropout_rate
        )


    def call(self, x, context):

        x = self.self_attention(x, x)
        x = self.cross_attention(x, context)
        x = self.feedforward(x)

        return x


class Decoder(tf.keras.layers.Layer):

    def __init__(self, n_blocks, n_heads, model_dim, ff_dim, dropout_rate):

        super().__init__()

        self.n_blocks = n_blocks
        self.decoder_blocks = [
            DecoderBlock(
                n_heads = n_heads, 
                model_dim = model_dim, 
                ff_dim = ff_dim, 
                dropout_rate = dropout_rate)
            for _ in range(n_blocks)
        ]

    
    def call(self, x, context):
        
        for i in range(self.n_blocks):
            x = self.decoder_blocks[i](x, context)

        return x


class MLP(tf.keras.layers.Layer):

    def __init__(self, hidden_dim, output_dim, dropout_rate):

        super().__init__()

        self.dense1 = tf.keras.layers.Dense(
            units = hidden_dim,
            activation = 'relu'
        )
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.concat = tf.keras.layers.Concatenate()
        self.dense2 = tf.keras.layers.Dense(
            units = output_dim,
            activation = 'relu'
        )
        self.dense3 = tf.keras.layers.Dense(
            units = output_dim,
            activation = 'tanh',
            kernel_regularizer = 'l2'
        )


    def call(self, inputs):

        x = self.dense1(inputs)
        x = self.dropout(x)
        x = self.concat([x, inputs])
        x = self.dense2(x)
        x = self.dense3(x)

        return x


class HybridModel(tf.keras.Model):

    def __init__(
            self, 
            token_embed_dim, 
            conv_dim, 
            filter_size, 
            dropout_rate, 
            n_blocks, 
            n_heads, 
            ff_dim, 
            trans_dropout_rate,
            dense1_dim,
            dense2_dim,
            dense3_dim,
            hidden_dim, 
            output_dim, 
            mlp_dropout_rate,
            seq_only = False
    ):
        '''
        Parameters:
        -----------
        token_embed_dim: The embedding dimension of the token embedding layer.
        conv_dim: The number of filters in the convolution layers. It should be equal to the 
            number of units in the second dense layer of feed forward network, i.e., the
            output dimension of ffn.
        filter_size: The filter size of the convolution layers.
        #pool_size: The pool size of the pooling layer.
        dropout_rate: The dropout rate of the dropout layer.
        n_blocks: The number of encoder and decoder blocks.
        n_heads: The number of heads in the multi-head attention.
        ff_dim: The number of units in the first dense layer of the feed forward network.
        trans_dropout_rate: The dropout rate in the transformer.
        dense1_dim: The number of units in the first dense layer for combined sequence and 
            non-sequence features.
        dense2_dim: The number of units in the second dense layer for combined sequence and 
            non-sequence features.
        dense3_dim: The number of units in the third dense layer for combined sequence and 
            non-sequence features.
        hidden_dim: The number of units in the hidden layer of the MLP.
        output_dim: The number of units in the output layer of the MLP.
        mlp_dropout_rate: The dropout rate in the MLP.
        seq_only: Whether to only use sequence features. If False, both sequence and non-sequence 
            features will be used.
        '''
        
        super().__init__()

        self.token_embedding = tf.keras.layers.Embedding(
            input_dim = 30, 
            output_dim = token_embed_dim,
            name = 'seq_embedding'
        )
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim = 30,
            output_dim = conv_dim,
            name = 'pos_embedding'
        )
        self.seq_conv = tf.keras.layers.Conv1D(   
            filters = conv_dim,
            kernel_size = filter_size,
            padding = 'same',
            activation = 'relu',
            name = 'seq_conv'
        )
        self.seq_dropout = tf.keras.layers.Dropout(
            rate = dropout_rate,
            name = 'seq_dropout'
        )
        self.comb_conv1 = tf.keras.layers.Conv1D(   
            filters = conv_dim,
            kernel_size = filter_size,
            padding = 'same',
            activation = 'relu',
            name = 'combine_conv1'
        )
        self.comb_conv2 = tf.keras.layers.Conv1D(   
            filters = conv_dim,
            kernel_size = filter_size,
            padding = 'same',
            activation = 'relu',
            name = 'combine_conv2'
        )
        self.encoder = Encoder(
            n_blocks = n_blocks,
            n_heads = n_heads,
            model_dim = conv_dim,
            ff_dim = ff_dim,
            dropout_rate = trans_dropout_rate,
        )
        self.decoder = Decoder(
            n_blocks = n_blocks,
            n_heads = n_heads,
            model_dim = conv_dim,
            ff_dim = ff_dim,
            dropout_rate = trans_dropout_rate,
        )
        self.comb_flatten = tf.keras.layers.Flatten(name = 'combine_flatten')
        self.trans_flatten = tf.keras.layers.Flatten(name = 'transformer_flatten')
        self.mlp = MLP(
            hidden_dim = hidden_dim, 
            output_dim = output_dim, 
            dropout_rate = mlp_dropout_rate
        )
        self.concat = tf.keras.layers.Concatenate(name = 'concate')
        self.seq_nonseq_dense1 = tf.keras.layers.Dense(
            units = dense1_dim,
            activation = 'relu',
            kernel_regularizer = 'l2',
            bias_regularizer = 'l2',
            name = 'seq_nonseq_dense1'
        )
        self.seq_nonseq_dropout1 = tf.keras.layers.Dropout(
            rate = dropout_rate,
            name = 'seq_nonseq_dropout1'
        )
        self.seq_nonseq_dense2 = tf.keras.layers.Dense(
            units = dense2_dim,
            activation = 'relu',
            kernel_regularizer = 'l2',
            bias_regularizer = 'l2',
            name = 'seq_nonseq_dense2'
        )
        self.seq_nonseq_dropout2 = tf.keras.layers.Dropout(
            rate = dropout_rate,
            name = 'seq_nonseq_dropout2'
        )
        self.seq_nonseq_dense3 = tf.keras.layers.Dense(
            units = dense3_dim,
            activation = 'relu',
            name = 'seq_nonseq_dense3'
        )
        self.seq_nonseq_dropout3 = tf.keras.layers.Dropout(
            rate = dropout_rate,
            name = 'seq_nonseq_dropout3'
        )
        self.seq_nonseq_dense4 = tf.keras.layers.Dense(
            units = 1,
            name = 'seq_nonseq_dense4'
        )

        self.seq_only = seq_only


    def call(self, inputs):
        '''
        Parameters:
        -----------
        inputs: A list of two tensors [seq, nonseq], where the first tensor is the sequence 
            features with shape (n_samples, sequence_len) and the second tensor is the non-sequence 
            features with shape (n_samples, n_features).
        '''

        if self.seq_only:
            seq = inputs
        else:
            seq, nonseq = inputs
        
        seq_embedded = self.token_embedding(seq)   
        seq_conv = self.seq_conv(seq_embedded)      
        seq_pool = seq_conv   
        
        pos = np.arange(1, seq.shape[-1]+1)
        pos_embedded = self.position_embedding(pos)   

        comb_conv1 = self.comb_conv1(seq_pool + pos_embedded)   
        comb_flattened = self.comb_flatten(comb_conv1)   

        seq_drop = self.seq_dropout(seq_pool)   
        comb_conv2 = self.comb_conv2(seq_drop + pos_embedded)   
        
        enc_output = self.encoder(comb_conv2)   
        dec_output = self.decoder(comb_conv1, enc_output)   

        trans_flattened = self.trans_flatten(dec_output)   

        if self.seq_only:
            comb_final = self.concat([0.2*comb_flattened, 0.8*trans_flattened])
        else:
            nonseq_mlp = self.mlp(nonseq)
            comb_final = self.concat([0.2*comb_flattened, 0.8*trans_flattened, nonseq_mlp])

        comb_final_dense1 = self.seq_nonseq_dense1(comb_final)
        comb_final_drop1 = self.seq_nonseq_dropout1(comb_final_dense1)
        comb_final_dense2 = self.seq_nonseq_dense2(comb_final_drop1)
        comb_final_drop2 = self.seq_nonseq_dropout2(comb_final_dense2)
        comb_final_dense3 = self.seq_nonseq_dense3(comb_final_drop2)
        comb_final_drop3 = self.seq_nonseq_dropout3(comb_final_dense3)
        output = self.seq_nonseq_dense4(comb_final_drop3)

        return output
    

    def build_graph(self):

        seq_inputs = tf.keras.layers.Input(shape = (23), name = 'seq_input')

        if self.seq_only:
            model = tf.keras.Model(
                inputs = seq_inputs, 
                outputs = self.call(seq_inputs)
            )

        else:
            nonseq_inputs = tf.keras.layers.Input(shape = (9), name = 'nonseq_input')

            model = tf.keras.Model(
                inputs = [seq_inputs, nonseq_inputs], 
                outputs = self.call([seq_inputs, nonseq_inputs])
            )

        return model
