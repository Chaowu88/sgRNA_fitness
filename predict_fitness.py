'''Predict fitness of the selected sgRNA targets.
'''


__author__ = 'Chao Wu'


import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import load


OUT_FILE = 'predicted_fitness_full_length.csv'
MODEL_DIR = 'hybrid'
DATA_FILE = 'prediction_data_full_length.csv'
CAT_SCALER_FILE = 'nonseq_cat.scaler'
NUM_SCALER_FILE = 'nonseq_num.scaler'
TARGET_SCALER_FILE = 'target.scaler'


def main():

    cat_scaler = load(CAT_SCALER_FILE)
    num_scaler = load(NUM_SCALER_FILE)
    target_scaler = load(TARGET_SCALER_FILE)

    data = pd.read_csv(DATA_FILE, header = 0, index_col = None)

    X_seq = data['seq'].values
    X_seq = tf.keras.layers.TextVectorization(
        split = 'character', 
        vocabulary = ['a', 't', 'c', 'g']   
    )(X_seq).numpy()

    X_nonseq_cat = data[['essential', 'ori', 'coding']].astype(object)
    X_nonseq_cat = cat_scaler.transform(X_nonseq_cat).toarray()

    X_nonseq_num = data['pos'].values.reshape((-1, 1))
    X_nonseq_num = num_scaler.transform(X_nonseq_num)

    X_nonseq = np.concatenate((X_nonseq_cat, X_nonseq_num), axis = 1)

    model = tf.keras.saving.load_model(MODEL_DIR)   
    
    Y_pred = model.predict([X_seq, X_nonseq])
    fitness = target_scaler.inverse_transform(Y_pred)

    data['fitness'] = fitness
    data.to_csv(OUT_FILE, header = True, index = False)




if __name__ == '__main__':

    main()