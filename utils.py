'''
MIT License

Copyright (c) 2022 Robin Cugny, IRIT and SolutionData Group, <robin.cugny@irit.fr>
Copyright (c) 2022 Julien Aligon, IRIT, <julien.aligon@irit.fr>
Copyright (c) 2022 Max Chevalier, IRIT, <max.chevalier@irit.fr>
Copyright (c) 2022 Geoffrey Roman Jimenez, SolutionData Group, <groman-jimenez@solutiondatagroup.fr>
Copyright (c) 2022 Olivier Teste, IRIT, <olivier.teste@irit.fr>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import  numpy as np
import pandas as pd
import pickle

questions_to_xai_sol = {'Why':['LIME','SHAP'],
                        'What':['MMD','Protodash','kmedoids']}

#TODO use dict above to populate list below
xai_sol = ['LIME','SHAP','MMD','Protodash','kmedoids']

hpo_list = ["random", "gp"]

#TODO add discretize continuous
#TODO change for a data class
hp_possible_values = {
    'LIME':
        {
            'num_samples':[10,10000]
        },
    'SHAP':{
        'summarize':["KernelExplainer","Sampling"],
        'nsamples':[10,2048],
        #TODO fix float giving robustness of 0
        # 'l1_reg':["num_features(int)", "auto", "aic", "bic", "float"],
        'l1_reg':["num_features(int)", "auto", "aic", "bic"],
    },
    'MMD':{
          'gamma':[0,1],
        #   'ktype':[0,1]
        },
    'Protodash':{
          'kernelType':['Gaussian','other'],
          'sigma':[0,5]
        },
    'kmedoids':{
          'metric':["euclidean","manhattan","cosine"],
          'method':["alternate","pam"],
          'init':["random", "heuristic", "k-medoids++", "build"],
          'max_iter':[0,300]
        }

}

def reorder_attributes(att, feature_names):
    """Sort the feature importances explanations using the same ordrer as the features in data.
    They are produced in order of importance which prevent from comparing them with each other. 

    Parameters
    ----------
    att : dict
        Feature importance explanation with feature name as key and importance as value.
    feature_names : list
        Names of the features as in the dataset.

    Returns
    -------
    list
        Feature importance explanation as a sorted list.
    """        
    return [att[f] for f in feature_names if f in att.keys()]

#TODO More genericity in data types
def load_dataset(path, label):
    """Loads a local file to use it as a dataset and use label as the target for the model.

    Parameters
    ----------
    path : str
        Path of the file.
    label : str
        Name of the target "feature".

    Returns
    -------
    ndarray, ndarray, list
        X the dataset, y the label column, feature_names.
    """    
    X = pd.read_csv(path)
    y = X.pop(label)
    feature_names = X.columns
    X = X.to_numpy()
    y = y.to_numpy()
    return X, y, feature_names

def load_model(path):
    """Loads a local file as the model with pickle.

    Parameters
    ----------
    path : str
        Path of the file.

    Returns
    -------
    model (sklearn)
        The model to explain.
    """    
    m = pickle.load(open(path, "rb"))
    return m
