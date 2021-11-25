import numpy as np
from tqdm import tqdm
from skopt import gp_minimize
from functools import partial
from XAI_solutions import set_up_explainer, get_local_exp
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import warnings
warnings.filterwarnings('ignore')

def lipschitz_ratio(x, y, function, reshape=None, minus=False):

    # Need this ugly hack because skopt sends lists
    if type(x) is list:
        x = np.array(x)
    if type(y) is list:
        y = np.array(y)
    if type(function(x)) is list:
        fx = np.array(function(x))
    else:
        fx = function(x)
    if type(function(y)) is list:
        fy = np.array(function(y))
    else:
        fy = function(y)

    if reshape is not None:
        # Necessary because gpopt requires to flatten things, need to restrore expected sshape here
        x = x.reshape(reshape)
        y = y.reshape(reshape)
    #print(x.shape, x.ndim)
    multip = -1 if minus else 1

    return multip * np.linalg.norm(fx - fy) / np.linalg.norm(x - y)

def compute_lipschitz_robustness(xai_sol, parameters, context):
    X=context["X"]
    verbose=context["verbose"]

    eps = 0.1
    njobs = -1
    if xai_sol in ['LIME','SHAP']:
        n_calls = 10
    else:
        n_calls = 100

    def exp(x):
        return get_local_exp(xai_sol, x, parameters, context)

    list_lip = []
    x_opts = []

    for i in tqdm(range(2)):
    # for i in tqdm(range(len(X))):
        x = X[i]
        orig_shape = x.shape
        lwr = (x - eps).flatten()
        upr = (x + eps).flatten()
        bounds = list(zip(*[lwr, upr]))
        f = partial(lipschitz_ratio, x, function=exp,
                    reshape=orig_shape, minus=True)
        res = gp_minimize(f, bounds, n_calls=n_calls,
                            verbose=verbose, n_jobs=njobs)
        lip, x_opt = -res['fun'], np.array(res['x'])
        list_lip.append(lip)
        x_opts.append(x_opt)
    
    score = np.mean(list_lip)
    return score

def compute_infidelity(xai_sol, parameters, context):
    X = context["X"]
    model = context['model']
    eps = 0.1
    list_inf = []
    for i in tqdm(range(2)):
    # for i in tqdm(range(len(X))):
        x = X[i]
        pertubation_diff = []
        for j in range(2):
            x0 = x + np.random.rand(len(x))*2*eps-eps
            exp = get_local_exp(xai_sol, x, parameters, context)
            # exp0 = get_local_exp(xai_sol, x, parameters, context)
            exp_x = np.matmul(x,np.asarray(exp).T)
            exp_x0 = np.matmul(x0,np.asarray(exp).T)
            pred_x = model.predict(x.reshape(1, -1))[0]
            pred_x0 = model.predict(x0.reshape(1, -1))[0]
            pertubation_diff.append((exp_x-exp_x0-(pred_x-pred_x0))**2)
        list_inf.append(np.mean(pertubation_diff))
    score = np.mean(list_inf)
    return score

def evaluate(xai_sol, parameters, property, context):
    # Set up of XAI solutions before computing evaluation
    if xai_sol in ['LIME','SHAP']:
        context['explainer'] = set_up_explainer(xai_sol, parameters, context)
    
    # Computing evaluation for specified property
    if property == 'robustness':
        if context['question']=="Why":
            score = compute_lipschitz_robustness(xai_sol, parameters, context)
    if property == 'fidelity':
        if context['question']=="Why":
            score = compute_infidelity(xai_sol, parameters, context)
    return score

def linear_scalarization(score_hist, properties_list, context):
    scaling = context["scaling"]
    weights = context["weights"]

    score_hist["aggregated_score"] = np.zeros(len(score_hist["aggregated_score"])+1)

    for i,property in enumerate(properties_list):
        if len(score_hist[property])>1:
            if scaling == "MinMax":
                scaler = MinMaxScaler()
            if scaling == "Std":
                scaler = StandardScaler()
            score_hist["scaled_"+property] = scaler.fit_transform(np.asarray(score_hist[property]).reshape(-1, 1)).reshape(1, -1).tolist()[0]
            # print("debug")
            # print(score_hist[property])
            # print(np.asarray(score_hist["scaled_"+property]))
            # print(weights[i])
            # print(len(properties_list))
            # print("-----")
            # print(np.asarray(score_hist["scaled_"+property]) * weights[i]/len(properties_list))
            # print(score_hist["aggregated_score"])
            score_hist["aggregated_score"] += np.asarray(score_hist["scaled_"+property]).reshape(score_hist["aggregated_score"].shape) * weights[i]/len(properties_list)
        else :
            score_hist["scaled_"+property] = 0

    score_hist["aggregated_score"]=list(score_hist["aggregated_score"])

    return score_hist