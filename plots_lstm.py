import pickle
import numpy as np 
import pandas as pd 
import seaborn as sns
import scipy.stats as ss
from matplotlib import pyplot as P

def plot_predicted_vs_data(predicted, Ydata, indice, label, pred_window, factor, split_point=None, uncertainty=False, label_name = 'predict'):
    """
    Plot the model's predictions against data
    :param predicted: model predictions
    :param Ydata: observed data
    :param indice:
    :param label: Name of the locality of the predictions
    :param pred_window:
    :param factor: Normalizing factor for the target variable
    """

    P.figure()
    if len(predicted.shape) == 2:
        df_predicted = pd.DataFrame(predicted).T
        df_predicted25 = None
    else:
        df_predicted = pd.DataFrame(np.percentile(predicted, 50, axis=2))
        df_predicted25 = pd.DataFrame(np.percentile(predicted, 2.5, axis=2))
        df_predicted975 = pd.DataFrame(np.percentile(predicted, 97.5, axis=2))
        uncertainty = True
    ymax = max(predicted.max() * factor, Ydata.max() * factor)

    if split_point != None: 
        if split_point == len(Ydata):

            P.vlines(indice[-1], 0, ymax, "g", "dashdot", lw=2, label = 'Train/Test')

        else:
            P.vlines(indice[split_point + 7], 0, ymax, "g", "dashdot", lw=2, label = 'Train/Test')
        
    #P.text(indice[split_point + 2], 0.6 * ymax, "Out of sample Predictions")
    # plot only the last (furthest) prediction point
    P.plot(indice[len(indice)-Ydata.shape[0]:], Ydata[:, -1] * factor, 'k-', alpha=0.7, label='data')
    P.plot(indice[len(indice)-Ydata.shape[0]:], df_predicted.iloc[:,-1] * factor, 'r-', alpha=0.5, label='median')
    if uncertainty:
        P.fill_between(indice[7:], df_predicted25[df_predicted25.columns[-1]] * factor,
                       df_predicted975[df_predicted975.columns[-1]] * factor,
                       color='b', alpha=0.3)

    # plot all predicted points
    # P.plot(indice[pred_window:], pd.DataFrame(Ydata)[7] * factor, 'k-')
    # for n in range(df_predicted.shape[1] - pred_window):
    #     P.plot(
    #         indice[n: n + pred_window],
    #         pd.DataFrame(Ydata.T)[n] * factor,
    #         "k-",
    #         alpha=0.7,
    #     )
    #     P.plot(indice[n: n + pred_window], df_predicted[n] * factor, "r-")
    #     try:
    #         P.vlines(
    #             indice[n + pred_window],
    #             0,
    #             df_predicted[n].values[-1] * factor,
    #             "b",
    #             alpha=0.2,
    #         )
    #     except IndexError as e:
    #         print(indice.shape, n, df_predicted.shape)
    tag = '_unc' if uncertainty else ''
    P.grid()
    #P.title("Predictions for {}".format(label))
    P.xlabel("time")
    P.ylabel("incidence")
    P.xticks(rotation=45)
    P.legend()
    P.savefig(f'./plots/lstm/{label_name}.png',bbox_inches='tight',  dpi = 300)
    P.show()
    
    return 


def plot_transf_predicted_vs_data(df_predicted_t, df_predicted, Ydata, indice, label, pred_window, factor, split_point=None, uncertainty=False, label_name = 'transf', label1 = 'transf', label2 = 'chik data'):
    """
    Plot the model's predictions against data
    :param predicted: model predictions
    :param Ydata: observed data
    :param indice:
    :param label: Name of the locality of the predictions
    :param pred_window:
    :param factor: Normalizing factor for the target variable
    """

    P.clf()
        
    uncertainty = True
    ymax = max( max(df_predicted.max()) * factor, Ydata.max() * factor, max(df_predicted_t.max()) * factor)

    if split_point != None:
        P.vlines(indice[split_point], 0, ymax, "g", "dashdot", lw=2, label = 'Train/Test')
    #P.text(indice[split_point + 2], 0.6 * ymax, "Out of sample Predictions")
    # plot only the last (furthest) prediction point
    P.plot(indice[len(indice)-Ydata.shape[0]:], Ydata[:, -1] * factor, 'k-', alpha=0.7, label='data')
    P.plot(indice[len(indice)-Ydata.shape[0]:], df_predicted.iloc[:,-1] * factor, 'r-', alpha=0.5, label=label1)
    P.plot(indice[len(indice)-Ydata.shape[0]:], df_predicted_t.iloc[:,-1] * factor, 'g-', alpha=0.5, label=label2)
   
                                  
    tag = '_unc' if uncertainty else ''
    P.grid()
    #P.title("Predictions for {}".format(label))
    P.xlabel("time")
    P.ylabel("incidence")
    P.xticks(rotation=30)
    P.legend()
    P.savefig(f'plots/lstm/{label_name}.png',bbox_inches='tight',  dpi = 300)
    P.show()

def predicted_vs_observed(predicted, real, city, state, doenca, model_name, city_name, plot=True):
    """
    Plot QQPlot for prediction values
    :param plot: generates an saves the qqplot when True (default)
    :param predicted: Predicted matrix
    :param real: Array of target_col values used in the prediction
    :param city: Geocode of the target city predicted
    :param state: State containing the city
    :param look_back: Look-back time window length used by the model
    :param all_predict_n: If True, plot the qqplot for every week predicted
    :return:
    """
    # Name = get_city_names([city])
    # data = get_alerta_table(city, state, doenca=doenca)

    obs_preds = np.hstack((predicted, real))
    q_p = [ss.percentileofscore(obs_preds, x) for x in predicted]
    q_o = [ss.percentileofscore(obs_preds, x) for x in real]
    plot_cross_qq(city, doenca, q_o, q_p, model_name, city_name)
    return np.array(q_o), np.array(q_p)


def plot_cross_qq(city, doenca, q_o, q_p,model_name, city_name):
    ax = sns.kdeplot(q_o[len(q_p) - len(q_o):], q_p, shade=True)
    ax.set_xlabel('observed')
    ax.set_ylabel('predicted')
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    P.plot([0, 100], [0, 100], 'k')
    #P.title(f'Transfer prediction percentiles with {model_name.lower()} for {doenca} at {city_name}')
    P.savefig(f'plots/lstm/cross_qqbplot_{model_name}_{doenca}_{city}.png', dpi=300)
