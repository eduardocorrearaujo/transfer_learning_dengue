This repo was created to store the codes used to create the results shown in the article (XXXXX). 

This study is an updated version of this work (available in pre-print): https://www.medrxiv.org/content/10.1101/2020.02.03.20020164v1.full.pdf

In this work, we applied the transfer learning technique to predict the chikungunya cases using a model trained to predict the dengue cases. We applied this methodology to four Brazilian cities: 

* Recife - PE;
* João Pessoa - PB; 
* Rio de Janeiro - RJ;
* Fortaleza - CE. 

Our repo is organized as follows:

* In the `data` folder, we have `.csv` tables with the data used to train and apply the models;
* In the `notebooks` folder, we have a jupyter notebook with the results of our analysis for each city. The name of the notebook is the city name;
* In the `plots` folder, we saved the figures;
* In the `predictions` folder, we save the predictions of the modes in `.pkl` files; 
* In the `saved_models` folder, we save the models trained. 
* In the `lstm.py` file, we have the definition of the functions used to apply the neural networks models used. 
* In the `pgbm.py` file, we have the definition of the functions used to apply the grandient boosting tree models. 
* In the `plots_lstm.py` file, we have the definition of the functions used to plot the results of the neural networks models used. 
* In the `plots_pgbm.py` file, we have the definition of the functions used to plot the results of the gradient boosting tree models. 
In the `preprocessing.py` file, we have the definition of the functions use to process the data saved in the `data` folder and transform it in the format accepted by the models in `lstm.py` and `pgbm.py`. 
