import os
import pickle
import joblib
import numpy as np
import pandas as pd 
from preprocessing import get_ml_data
from plots_pgbm import plot_prediction
from pgbm_nb import PGBMRegressor,PGBM
from preprocessing import build_lagged_features 
from sklearn.model_selection import train_test_split

def qf_prediction(city, state, predict_n, look_back, doenca = 'dengue', ratio = 0.75, ini_date = None, 
                  end_train_date = None, end_date = None):
    """
    Train a model for a single city and disease.
    :param city:
    :param state:
    :param predict_n:
    :param look_back:
    :return:
    """
 
    X_data, X_train, targets, target = get_ml_data(city, doenca, ini_date = ini_date, end_train_date = end_train_date, end_date = end_date, 
                                        ratio = ratio , predict_n = predict_n, look_back = look_back)

    preds = np.empty((len(X_data), predict_n))
    preds25 = np.empty((len(X_data), predict_n))
    preds975 = np.empty((len(X_data), predict_n))

    for d in range(1, predict_n + 1):
        tgt = targets[d][:len(X_train)]

        model = PGBMRegressor(n_estimators= 100,  distribution='poisson')
        
        model.fit(X_train, tgt)

        model.save(f'./saved_models/pgbm/{city}_{doenca}_city_model_{d}_pgbm.pt')

        pred = model.predict(X_data[:len(targets[d])].values)
        
        pred_dist = model.predict_dist(X_data[:len(targets[d])].values)
        
        pred25 = pred_dist.max(axis=0)
        pred = pred
        pred975 = pred_dist.min(axis=0)
        dif = len(X_data) - len(pred)
        if dif > 0:
            pred = list(pred) + ([np.nan] * dif)
            pred25 = list(pred25) + ([np.nan] * dif)
            pred975 = list(pred975) + ([np.nan] * dif)
        preds[:, (d - 1)] = pred
        preds25[:, (d - 1)] = pred25
        preds975[:, (d - 1)] = pred975

    x, y, y25, y975 = plot_prediction(preds, preds25, preds975, target, f'Predictions for {city}', state, len(X_train), doenca,
                                    label = f'pgbm_{city}')

    with open(f'./predictions/pgbm/pgbm_{city}_{doenca}_predictions.pkl', 'wb') as f:
        pickle.dump({'target':target,'dates': x, 'preds': y, 'preds25': y25,
                    'preds975': y975, 'train_size': len(X_train)
                    }, f)

    return preds, preds25, preds975, X_train, targets 


def cross_dengue_chik_prediction(city, state, predict_n, look_back, ini_date = '2020-01-01', end_date = None):
    """
    Functio to apply a model trained with dengue data in chik data. 
    """
    X_data, X_train, targets, target = get_ml_data(city, doenca = 'chik', ini_date = ini_date, end_date = end_date, 
                                        ratio = 0.99, predict_n = predict_n, look_back = look_back)

    preds = np.empty((len(X_data), predict_n))
    preds25 = np.empty((len(X_data), predict_n))
    preds975 = np.empty((len(X_data), predict_n))

    for d in range(1, predict_n + 1):

        model = PGBM()
        
        model.load(f'./saved_models/pgbm/{city}_dengue_city_model_{d}_pgbm.pt')

        pred = model.predict(X_data[:len(targets[d])].values)
        
        pred_dist = model.predict_dist(X_data[:len(targets[d])].values)
        
        pred25 = pred_dist.max(axis=0)
        pred975 = pred_dist.min(axis=0)

        dif = len(X_data) - len(pred)
        if dif > 0:
            pred = list(pred) + ([np.nan] * dif)
            pred25 = list(pred25) + ([np.nan] * dif)
            pred975 = list(pred975) + ([np.nan] * dif)
        preds[:, (d - 1)] = pred
        preds25[:, (d - 1)] = pred25
        preds975[:, (d - 1)] = pred975
        
    x, y, y25, y975 = plot_prediction(preds, preds25, preds975, target, 'Predictions for chik at ' + str(city) + ' applying the dengue model', state, None, 'chik',
                                    label = f'pgbm_cross_pred_{city}') 

    with open(f'./predictions/pgbm/pgbm_{city}_chik_cross_predictions.pkl', 'wb') as f:
        pickle.dump({'target': target, 'dates': x, 'preds': y, 'preds25': y25,
                    'preds975': y975, 'train_size': len(X_data)
                    }, f)

    return preds, preds25, preds975, X_data, target
