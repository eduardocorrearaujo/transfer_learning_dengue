import pickle
import numpy as np
import pandas as pd 
from time import time 
import tensorflow as tf 
import tensorflow.keras as keras
from preprocessing import get_nn_data
from tensorflow.keras.layers import LSTM
from tensorflow.keras import backend as K
from plots_lstm import plot_predicted_vs_data
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.metrics import mean_absolute_error, explained_variance_score, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score

def evaluate(model, Xdata, uncertainty=True):
    if uncertainty:
        predicted = np.stack([model.predict(Xdata, batch_size=1, verbose=0) for i in range(100)], axis=2)
    else:
        predicted = model.predict(Xdata, batch_size=1, verbose=0)
    return predicted

def calculate_metrics(pred, y_true, factor):
    """
    :param pred:
    :param y_true:
    :param factor:
    """
    metrics = pd.DataFrame(
        index=(
            "mean_absolute_error",
            "explained_variance_score",
            "mean_squared_error",
            "mean_squared_log_error",
            "median_absolute_error",
            "r2_score",
        )
    )
    for col in range(pred.shape[1]):
        y = y_true[:, col] * factor
        p = pred[:, col] * factor
        l = [
            mean_absolute_error(y, p),
            explained_variance_score(y, p),
            mean_squared_error(y, p),
            mean_squared_log_error(y, p),
            median_absolute_error(y, p),
            r2_score(y, p),
        ]
        metrics[col] = l
    return metrics

def custom_loss_msle(p = 1):
    """
    :param p: 
    """
    
    def my_loss_msle(y_true, y_pred):
        """
        :param y_true:
        :param y_pred: 
        """

        def f1(): 
            
            loss = tf.math.log(tf.math.add(y_true, 1)/ tf.math.add(y_pred, 1))
    
            loss = tf.square(loss)
        
            return tf.multiply(loss, p) 
        
        def f2(): 
            
            loss = tf.math.log(tf.math.add(y_true, 1)/ tf.math.add(y_pred, 1))
    
            loss = tf.square(loss)
        
            return loss

        msle = tf.cond(tf.less(tf.gather(y_true, 0)[1],tf.gather(y_true, 0)[3]) , 
                                     true_fn = f1,
                                     false_fn = f2 )

        return tf.reduce_mean(msle) 
    
    return my_loss_msle 

def build_model(hidden, features, predict_n, look_back=10, batch_size=1, loss = 'msle'):
    """
    Builds and returns the LSTM model with the parameters given
    :param hidden: int.number of hidden nodes
    :param features: int. Number of the features used to train the model
    :param predict_n: int.Number of observations that will be forecast (multi-step forecast)
    :param look_back: int.Number of time-steps to look back before predicting
    :param batch_size: int. batch size for batch training
    :param loss: string or function. Loss function to be used in the training process. 
    :return:
    """
    inp = keras.Input(
        shape=(look_back, features),
        # batch_shape=(batch_size, look_back, features)
    )
    
    x = Bidirectional(LSTM(
        hidden,
        input_shape=(look_back, features),
        stateful=False,
        batch_input_shape=(batch_size, look_back, features),
        return_sequences=True,
        #activation='relu',
        dropout=0.1,
        recurrent_dropout=0.1,
        implementation=2,
        unit_forget_bias=True,
    ), merge_mode = 'ave', name = 'bidirectional_1')(inp, training=True)     

    x = Dropout(0.2, name='dropout_1')(x, training=True)
    
    x = LSTM(
        hidden,
        input_shape=(look_back, features),
        stateful=False,
        batch_input_shape=(batch_size, look_back, features),
        return_sequences= False,
        #activation='relu',
        dropout=0.1,
        recurrent_dropout=0.1,
        implementation=2,
        unit_forget_bias=True, name='lstm_1'
    )(x, training=True)

    x = Dropout(0.2, name = 'dropout_2')(x, training=True)

    out = Dense(
        predict_n,
        activation="relu",
        kernel_initializer="random_uniform",
        bias_initializer="zeros",
        name = 'dense'
    )(x)
    model = keras.Model(inp, out)

    start = time()
    model.compile(loss = loss, optimizer="adam", metrics=["accuracy", "mape", "mse"])
    print("Compilation Time : ", time() - start)
    plot_model(model, to_file="LSTM_model.png")
    print(model.summary())
    return model


def train(model, X_train, Y_train, label, batch_size=1, epochs=10, geocode=None, overwrite=True, validation_split = 0.25):
    """
    Train the lstm model 
    :param model: LSTM model compiled and created with the build_model function 
    :param X_train: array. Arrays with the features to train the model 
    :param Y_train: array. Arrays with the target to train the model
    :param label: string. Name to be used to save the model
    :param batch_size: int. batch size for batch training
    :param epochs: int.  Number of epochs used in the train 
    :param geocode: int. Analogous to city (IBGE code), it will be used in the name of the saved model
    :param overwrite: boolean. If true we overwrite a saved model with the same name. 
    :param validation_split: float. The slice of the training data that will be use to evaluate the model 
    """
    
    TB_callback = TensorBoard(
        log_dir="./tensorboard",
        histogram_freq=0,
        write_graph=True,
        write_images=True,
        # embeddings_freq=10
    )

    hist = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        verbose=1,
        callbacks=[TB_callback, EarlyStopping(monitor = 'loss', patience=10)]
    )
    
    model.save(f"../saved_models/lstm/trained_{geocode}_model_{label}.h5", overwrite=overwrite)

    return model, hist


def make_pred(model, city, doenca,  epochs, ini_date = None, end_train_date = None, 
                 end_date = None,ratio= 0.75, hidden = 8,
                 predict_n = 4, look_back =  4,
                  label = 'model', filename = None):
    """
    :param model: tensorflow model. 
    :param city: int. IBGE code of the city. 
    :param state: string. 
    :param epochs: int. 
    :param ini_date: string or None.
    :param end_train_date: string or None.
    :param end_date: string or None.
    :param hidden: int.
    :param predict_n: int.
    :param look_back: int.
    :param label: string.
    """

    df,factor,  X_train, Y_train, X_pred, Y_pred = get_nn_data(city, doenca = doenca, ini_date = ini_date, 
                                                     end_date = end_date, end_train_date = end_train_date,
                                                    ratio = ratio, look_back = look_back,
                                                    predict_n = predict_n, filename = filename)
   
    model, hist =  train(model, X_train, Y_train, label = label, batch_size=1, epochs=epochs, geocode=city, overwrite=True, validation_split = 0.25)
   
    pred = evaluate(model, X_pred)

    df_pred = pd.DataFrame(np.percentile(pred, 50, axis=2))
    df_pred25 = pd.DataFrame(np.percentile(pred, 2.5, axis=2))
    df_pred975 = pd.DataFrame(np.percentile(pred, 97.5, axis=2))
    
    with open(f'../predictions/lstm/lstm_{city}_{doenca}_{label}.pkl', 'wb') as f:
        pickle.dump({'xdata': X_train, 'indice': list(df.index)  , 'target': Y_pred,  'pred': df_pred, 'ub': df_pred975,  'lb':df_pred25,
                    'factor': factor, 'city': city}, f)

    indice = list(df.index)
    indice = [i.date() for i in indice] 

    plot_predicted_vs_data(pred, Y_pred, indice, city, 4, factor, split_point=len(Y_train), uncertainty= True, 
                        label_name = f'{label}_{city}',
            )           

    metrics = calculate_metrics(np.percentile(pred, 50, axis=2), Y_pred, factor)

    return metrics  

def apply_dengue_chik(city_name, city, ini_date = '2021-01-01', 
                     end_date = '2022-01-01', look_back = 4,
                     predict_n = 4,  label_m = f'dengue_train_base', filename = None ): 

    """
    Function to apply a model trained with dengue data using chik data. 
    """

    df,factor,  X_train, Y_train, X_pred, Y_pred = get_nn_data(city, doenca = 'chik',
                                                    ini_date = ini_date, end_date = end_date,
                                                    end_train_date = None, ratio = 1,
                                                    look_back = look_back,
                                                    predict_n = predict_n,
                                                    filename = filename)
    
    model_dengue = keras.models.load_model(f'../saved_models/lstm/trained_{city}_model_{label_m}.h5',  compile =False)

    pred_chik = evaluate(model_dengue, X_pred)

    df_pred_chik = pd.DataFrame(np.percentile(pred_chik, 50, axis=2))
    df_pred25_chik = pd.DataFrame(np.percentile(pred_chik, 2.5, axis=2))
    df_pred975_chik = pd.DataFrame(np.percentile(pred_chik, 97.5, axis=2))

    with open(f'../predictions/lstm/lstm_{city}_dengue_model_chik_predictions.pkl', 'wb') as f:
        pickle.dump({'indice': list(df.index)  , 'target': Y_pred,  'pred': df_pred_chik, 'ub': df_pred975_chik,  
                     'lb':df_pred25_chik, 
                    'factor': factor, 'city_name': city_name
                    }, f)

    indice = list(df.index)
    indice = [i.date() for i in indice]

    plot_predicted_vs_data(pred_chik, Y_pred, indice, f'chik at {city_name} applying the dengue model', 4, factor, split_point= None, uncertainty= True, label_name = f'chik_{city_name.lower()}')

    metrics = calculate_metrics(np.percentile(pred_chik, 50, axis=2), Y_pred, factor)

    return metrics

def train_fine(doenca, model, X_train, Y_train, batch_size=1, epochs=10, geocode=None, overwrite=True):
    """
    Function to apply the fine tune step.
    """

    TB_callback = TensorBoard(
        log_dir="./tensorboard",
        histogram_freq=0,
        write_graph=True,
        write_images=True,
        # embeddings_freq=10
    )
    """
    Code to apply the transfer learning technique 
    """

    model.compile(loss='msle', optimizer= "adam", metrics=["accuracy", "mape", "mse"])

    hist = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        verbose=0,
        callbacks=[TB_callback, EarlyStopping(monitor = 'loss', patience=10)]
    )
    
    model.save(f"../saved_models/lstm/trained_{geocode}_model_{doenca}.h5", overwrite=overwrite)

    return model

def transf_model(filename = 'trained_2312908_model_dengue.h5', features = 11,  
                 predict_n = 4, look_back = 4, batch_size = 1, loss = 'msle'):
    """
    Function to load a model trained with dengue data, freeze some layers and retrain it with chik data 
    """
    
    # get the model trained with the dengue data 
    base_model = keras.models.load_model(filename, compile = False)
    
    base_model.compile(loss = loss, optimizer="adam", metrics=["accuracy", "mape", "mse"])
    
    base_model.trainable = False 
    
    inp = keras.Input(
        shape=(look_back, features))
    
    x = base_model.get_layer('bidirectional_1')(inp, training=False)
    
    x = base_model.get_layer('dropout_1')(x, training = True)
 
    x = base_model.get_layer('lstm_1')(x, training = False)
    
    x = Dropout(0.2, name = 'dropout_2')(x, training = True)

    out = Dense(
        predict_n,
        activation="relu",
        kernel_initializer="random_uniform",
        bias_initializer="zeros",
        name = 'dense'
    )(x)
    
    model = keras.Model(inp, out)
    
    start = time()
 
    model.compile(loss = loss, optimizer= "adam", metrics=["accuracy", "mape", "mse"])
    print("Compilation Time : ", time() - start)

    print(model.summary())
    return model 


def transf_chik_pred(city_name, city, ini_date = '2021-01-01', end_train_date = '2021-03-01',  
                            end_date = '2022-12-31', ratio =0.75, filename = 'trained_2312908_model_dengue.h5',  epochs =100, features = 11,  
                            predict_n = 4, look_back = 4, loss = 'msle', validation_split = 0.15,
                            label = f'transf_chik', data_filename = None): 

    """
    Function to apply the transfer learning loading a model trained with dengue data and retraining it using the chik data. 
    """


    
    df,factor,  X_train, Y_train, X_pred, Y_pred = get_nn_data(city, doenca = 'chik', ini_date = ini_date, 
                                                     end_date = end_date, end_train_date = end_train_date,
                                                    ratio = ratio, look_back = look_back,
                                                    predict_n = predict_n,
                                                    filename = data_filename)

    model_transf =  transf_model(filename = filename, features = features,  
                            predict_n = predict_n, look_back = look_back, batch_size = 1, loss = loss )


    model_transf, hist = train(model = model_transf, X_train = X_train, Y_train = Y_train, label = label,  epochs=epochs, geocode= city, overwrite=True,
         validation_split = validation_split)

    model_transf.trainable = True
     
    model_fine= train_fine(label, model_transf, X_train, Y_train, epochs=epochs, geocode= city, overwrite=True)

    pred = evaluate(model_fine, X_pred)

    df_pred = pd.DataFrame(np.percentile(pred, 50, axis=2))
    df_pred25 = pd.DataFrame(np.percentile(pred, 2.5, axis=2))
    df_pred975 = pd.DataFrame(np.percentile(pred, 97.5, axis=2))
    
    indice = list(df.index)
    indice = [i.date() for i in indice]
    
    with open(f'../predictions/lstm/tl_{city}_{label}.pkl', 'wb') as f:
        pickle.dump({'indice': indice, 'target': Y_pred,  'pred': df_pred,'pred25': df_pred25, 'pred975': df_pred975,                     
                    'factor': factor, 'city_name': city_name}, f)

    plot_predicted_vs_data(pred, Y_pred, indice, f'{city_name} - chik - transf model', 4, factor, split_point=len(Y_train), uncertainty= True, label_name = f'transf_{city_name.lower()}')
    
   
    return 