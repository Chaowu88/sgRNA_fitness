'''
Train a hybrid transformer model to predict the effect of gRNA on cell fitness represented by 
fold change in abundance (log2FC) of the guide RNA in the library (10.1038/s41467-018-04209-5). 
It is recommended to train on a platform with GPU support.
'''


__author__ = 'Chao Wu'


import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import tensorflow as tf
from joblib import dump
from model import HybridModel

DATA_FILE = 'training_data.csv'
MODEL_PARAMS = {
    'token_embed_dim': 80, 
    'conv_dim': 256, 
    'filter_size': 5, 
    'pool_size': 1, 
    'dropout_rate': 0.3, 
    'n_blocks': 4, 
    'n_heads': 8,
    'ff_dim': 128, 
    'trans_dropout_rate': 0.1,
    'dense1_dim': 128,
    'dense2_dim': 64,
    'dense3_dim': 32,
    'hidden_dim': 128, 
    'output_dim': 256, 
    'mlp_dropout_rate': 0.05,
    'learning_rate': 0.00002
}
EPOCHS = 100
BATCH_SIZE = 32
SEQ_ONLY = False
MODE = 'test'   # {'train', 'test'}


def prepare_dataset(data_file):

    data = pd.read_csv(data_file, header = 0, index_col = None)
    
    X_seq = data['seq'].values
    X_seq = tf.keras.layers.TextVectorization(
        split = 'character', 
        vocabulary = ['a', 't', 'c', 'g']   
    )(X_seq).numpy()
    
    X_nonseq_cat = data[['essential', 'ori', 'coding']]
    nonseq_cat_scaler = OneHotEncoder()
    nonseq_cat_scaler.fit(X_nonseq_cat)
    X_nonseq_cat = nonseq_cat_scaler.transform(X_nonseq_cat).toarray()
    dump(nonseq_cat_scaler, 'nonseq_cat.scaler')
    
    X_nonseq_num = data['pos'].values.reshape((-1, 1))
    nonseq_num_scaler = MinMaxScaler()
    nonseq_num_scaler.fit(X_nonseq_num)
    X_nonseq_num = nonseq_num_scaler.transform(X_nonseq_num)
    dump(nonseq_num_scaler, 'nonseq_num.scaler')
    
    X_nonseq = np.concatenate((X_nonseq_cat, X_nonseq_num), axis = 1)

    Y = data['fitness'].values.reshape((-1, 1))
    target_scaler = StandardScaler()
    target_scaler.fit(Y)
    Y = target_scaler.transform(Y)
    dump(target_scaler, 'target.scaler')
    
    return train_test_split(
        X_seq, 
        X_nonseq, 
        Y, 
        test_size = 0.25,   
        random_state = 1
    )


def build_model(
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
        lr
    ):
    
    model = HybridModel(
        token_embed_dim = token_embed_dim, 
        conv_dim = conv_dim, 
        filter_size = filter_size, 
        dropout_rate = dropout_rate, 
        n_blocks = n_blocks, 
        n_heads = n_heads,   
        ff_dim = ff_dim, 
        trans_dropout_rate = trans_dropout_rate,
        dense1_dim = dense1_dim,
        dense2_dim = dense2_dim,
        dense3_dim = dense3_dim,
        hidden_dim = hidden_dim, 
        output_dim = output_dim, 
        mlp_dropout_rate = mlp_dropout_rate,
        seq_only = SEQ_ONLY
    ).build_graph()

    model.compile(
        optimizer = tf.keras.optimizers.experimental.Adamax(learning_rate = lr), 
        loss = 'mse', 
        metrics = ['mse']   
    )
    
    return model


def main():

    (X_seq_train, 
     X_seq_test, 
     X_nonseq_train, 
     X_nonseq_test, 
     Y_train, 
     Y_test) = prepare_dataset(DATA_FILE)
    
    model = build_model(
        token_embed_dim = MODEL_PARAMS['token_embed_dim'], 
        conv_dim = MODEL_PARAMS['conv_dim'], 
        filter_size = MODEL_PARAMS['filter_size'], 
        dropout_rate = MODEL_PARAMS['dropout_rate'], 
        n_blocks = MODEL_PARAMS['n_blocks'], 
        n_heads = MODEL_PARAMS['n_heads'],    
        ff_dim = MODEL_PARAMS['ff_dim'], 
        trans_dropout_rate = MODEL_PARAMS['trans_dropout_rate'],
        dense1_dim = MODEL_PARAMS['dense1_dim'],
        dense2_dim = MODEL_PARAMS['dense2_dim'],
        dense3_dim = MODEL_PARAMS['dense3_dim'],
        hidden_dim = MODEL_PARAMS['hidden_dim'], 
        output_dim = MODEL_PARAMS['output_dim'], 
        mlp_dropout_rate = MODEL_PARAMS['mlp_dropout_rate'],
        lr = MODEL_PARAMS['learning_rate']
    )

    model_name = 'hybrid_seq_only' if SEQ_ONLY else 'hybrid'

    if MODE == 'train':
        stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5)
        model.fit(
            X_seq_train if SEQ_ONLY else [X_seq_train, X_nonseq_train], 
            Y_train, 
            epochs = EPOCHS, 
            batch_size = BATCH_SIZE, 
            validation_split = 0.2,   
            callbacks = [stopping],   
            verbose = 2
        )
        model.save(model_name)

    elif MODE == 'test':
        model = tf.keras.saving.load_model(model_name)
        Y_pred = model.predict(
            X_seq_test if SEQ_ONLY else [X_seq_test, X_nonseq_test]
        )
        
        print(f'Pearson r: {pearsonr(Y_pred.reshape(-1), Y_test.reshape(-1)).statistic}')
        print(f'Spearman r: {spearmanr(Y_pred.reshape(-1), Y_test.reshape(-1)).correlation}')
        print(f'R2: {r2_score(Y_test, Y_pred)}')




if __name__ == '__main__':

    main()