import  numpy as np
import pandas as pd
import pickle

questions_to_xai_sol = {'Why':['LIME','SHAP'],
                        'What':['MMD','Protodash','kmedoids']}
# questions_to_xai_sol = {'Why':['LIME']}
#TODO use dict above to populate list below
xai_sol = ['LIME','SHAP','MMD','Protodash','kmedoids']

hpo_list = ["random", "gp"]

#TODO add discretize continuous
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
        #   'kernelType':['Gaussian','other'],
          'kernelType':['other'],
          'sigma':[0,30]
        },
    'kmedoids':{
          'metric':["euclidean","manhattan","cosine"],
          'method':["alternate","pam"],
          'init':["random", "heuristic", "k-medoids++", "build"],
          'max_iter':[0,300]
        }

}

        # if xai_sol == 'MMD':
        #     parameters['gamma'] = None
        #     parameters['ktype'] = 0
        # if xai_sol == 'Protodash':
        #     parameters['kernelType'] = 'other'
        #     parameters['sigma'] = 2
        # if xai_sol == 'kmedoids':
        #     parameters['metric'] = "euclidean"
        #     parameters['method'] = "alternate"
        #     parameters['init'] = "heuristic"
        #     parameters['max_iter'] = 300


def reorder_attributes(att, feature_names):    
    return [att[f] for f in feature_names if f in att.keys()]

def load_dataset(path, label):
    X = pd.read_csv(path)
    y = X.pop(label)
    feature_names = X.columns
    X = X.to_numpy()
    y = y.to_numpy()
    return X, y, feature_names

def load_model(path):
    m = pickle.load(open(path, "rb"))
    return m
