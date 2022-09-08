'''
MIT License

Copyright (c) 2022 Robin Cugny, IRIT and SolutionData Group, <robin.cugny@irit.fr>

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

from matplotlib import docstring
import numpy as np
from tqdm import tqdm
from skopt import gp_minimize
from functools import partial
from XAI_solutions import set_up_explainer, get_local_exp, get_prototypes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import pairwise_distances
import pickle
import os

import warnings
warnings.filterwarnings('ignore') #OUCH

def lipschitz_ratio(x, y, function, reshape=None, minus=False):
    """
    Compute the ratio of the lipschitzian continuity for two points and a given function.
    
    Credits to David Alvarez-Melis and Tommi S. Jaakkola
        https://arxiv.org/abs/1806.08049
        https://github.com/dmelis/robust_interpret

    Parameters
    ----------
    x : list or numpy array
        First vector of a data point as a set of coordinates.
    y : list or np.array
        Second vector of a data point as a set of coordinates.
    function : callable
        Function that is evaluated, here it is the function that returns the explanation.
    reshape : tuple, optional
        [description], by default None
    minus : bool, optional
        [description], by default False

    Returns
    -------
    float
        Local ratio for the two data points.
    """
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
    """
    Computes the lipschitzian robustness score for a given XAI solution with the given 
    parameters on the context dataset and the context model.

    Parameters
    ----------
    xai_sol : str
        Name of the XAI solution that is evaluated.
    parameters : dict
        Parameters of the XAI solution for the current evaluation.
    context : dict
        Information of the context that may change the process.

    Returns
    -------
    float
        Lipschitzian robustness score (loss) for the XAI solution with the given parameters.
    """    
    es=context["ES"]
    IS=context["IS"]
    session_id = context["session_id"]

    X=context["X"]
    verbose=context["verbose"]
    eps = list(np.std(X, axis=0)*0.1)
    njobs = -1
    if xai_sol in ['LIME','SHAP']:
        n_calls = 10
    else:
        n_calls = 100

    def exp(x):
        return get_local_exp(xai_sol, x, parameters, context, update_order_feat=False)
    
    list_lip = []

    path = 'results/x_opts_'+xai_sol+session_id+'.p'
    if IS and os.path.exists(path):
        # print('Robustness uses previously computed points')
        x_opts = pickle.load(open(path, "rb"))
        for i in tqdm(range(len(x_opts))):
            if parameters['nfeatures']!=len(context["feature_names"]):
                get_local_exp(xai_sol, X[i], parameters, context)
            else:
                parameters['most_influent_features'] = list(np.arange(0,parameters['nfeatures']))
                
            if xai_sol=='LIME':
                context['explainer']=set_up_explainer(xai_sol, parameters, context)
            lip = lipschitz_ratio(X[i],x_opts[i],exp)
            list_lip.append(lip)

    else:
        x_opts = []
        stable_i=0
        # for i in tqdm(range(2)):
        for i in tqdm(range(len(X))):
            x = X[i]
            orig_shape = x.shape
            lwr = (x - eps).flatten()
            upr = (x + eps).flatten()
            bounds = list(zip(*[lwr, upr]))
            if parameters['nfeatures']!=len(context["feature_names"]):
                get_local_exp(xai_sol, X[i], parameters, context)
            else:
                parameters['most_influent_features'] = list(np.arange(0,parameters['nfeatures']))
            f = partial(lipschitz_ratio, x, function=exp,
                        reshape=orig_shape, minus=True)
            res = gp_minimize(f, bounds, n_calls=n_calls,
                                verbose=verbose, n_jobs=njobs)
            lip, x_opt = -res['fun'], np.array(res['x'])
            
            # TODO fix the pb with IS on LIME (is it even possible or important ?)
            if xai_sol=='LIME':
                context['explainer']=set_up_explainer(xai_sol, parameters, context)
                #Recalculate to have a normal initialization
                lip = lipschitz_ratio(X[i],x_opt,exp)

            list_lip.append(lip)
            # print("from gp_minimize",lip)
            # print("from recalculation",lipschitz_ratio(X[i],x_opt,exp))
            x_opts.append(x_opt)
            
            if es and abs(np.mean(list_lip[:-1])-np.mean(list_lip)) <= np.mean(list_lip)/10 and i>5:
                stable_i+=1
                if stable_i > 5:
                    break
            else:
                stable_i=0 

    if IS and not os.path.exists(path):
        pickle.dump(x_opts, open(path, "wb"))
    # print(x_opts)
    # print(list_lip)
    score = np.mean(list_lip)
    return -score

def compute_infidelity(xai_sol, parameters, context):
    """
    Computes the infidelity score for a given XAI solution with the given 
    parameters on the context dataset and the context model.

    Parameters
    ----------
    xai_sol : str
        Name of the XAI solution that is evaluated.
    parameters : dict
        Parameters of the XAI solution for the current evaluation.
    context : dict
        Information of the context that may change the process.

    Returns
    -------
    float
        Infidelity score (loss) for the XAI solution with the given parameters.
    """    
    es = context["ES"]
    IS = context["IS"]
    session_id = context["session_id"]
    X = context["X"]
    model = context['model']
    eps = list(np.std(X, axis=0)*0.1)
    nb_pert = 10
    list_inf = []

    path = 'results/perturb_infs_'+xai_sol+session_id+'.p'
    if IS and os.path.exists(path):
        # print('Fidelity uses previously computed points')
        perturb_infs = pickle.load(open(path, "rb"))
        for i in tqdm(range(len(perturb_infs))):
            x = X[i]
            pertubation_diff = []
            exp = get_local_exp(xai_sol, x, parameters, context)/context[xai_sol+"_std"]
            exp_x = np.matmul(x[parameters['most_influent_features']],np.asarray(exp).T)
            for j in range(nb_pert):
                x0 = perturb_infs[i]['x0'][j]
                # exp0 = get_local_exp(xai_sol, x0, parameters, context)[:parameters['nfeatures']]
                # exp_x0 = np.matmul(x0[:parameters['nfeatures']],np.asarray(exp0).T)
                exp_x0 = np.matmul(x0[parameters['most_influent_features']],np.asarray(exp).T)
                pred_x = perturb_infs[i]['pred_x'][j]
                pred_x0 = perturb_infs[i]['pred_x0'][j]
                # print("exp_x-exp_x0",exp_x-exp_x0)
                # print("pred_x-pred_x0",pred_x-pred_x0)
                pertubation_diff.append((exp_x-exp_x0-(pred_x-pred_x0))**2)
            list_inf.append(np.mean(pertubation_diff))
    else:
        # for i in tqdm(range(2)):
        perturb_infs = []
        stable_i=0
        for i in tqdm(range(len(X))):
            x = X[i]
            pertubation_diff = []
            pert = {'x0':[],'pred_x':[],'pred_x0':[]}
            exp = get_local_exp(xai_sol, x, parameters, context)/context[xai_sol+"_std"]
            exp_x = np.matmul(x[parameters['most_influent_features']],np.asarray(exp).T)
            for j in range(nb_pert):
                x0 = x + np.random.rand(len(x))*2*eps-eps
                # exp0 = get_local_exp(xai_sol, x0, parameters, context)[:parameters['nfeatures']]
                # exp_x0 = np.matmul(x0[:parameters['nfeatures']],np.asarray(exp0).T)
                exp_x0 = np.matmul(x0[parameters['most_influent_features']],np.asarray(exp).T)
                if context['task']=='regression':
                    pred_x = model.predict(x.reshape(1, -1))[0]
                    pred_x0 = model.predict(x0.reshape(1, -1))[0]
                else:
                    pred_x = max(model.predict_proba(x.reshape(1, -1))[0])
                    pred_x0 = max(model.predict_proba(x0.reshape(1, -1))[0])
                pertubation_diff.append((exp_x-exp_x0-(pred_x-pred_x0))**2)

                pert['x0'].append(x0)
                pert['pred_x'].append(pred_x)
                pert['pred_x0'].append(pred_x0)

            perturb_infs.append(pert)

            list_inf.append(np.mean(pertubation_diff))
            # print(" ES fid")
            # print(abs(np.mean(list_inf[:-1])-np.mean(list_inf)))
            # print(np.mean(list_inf)/10)
            if es and abs(np.mean(list_inf[:-1])-np.mean(list_inf)) <= np.mean(list_inf)/10 and i>5:
                stable_i+=1
                if stable_i > 5:
                    break
            else:
                stable_i=0 
    if IS and not os.path.exists(path):
        pickle.dump(perturb_infs, open(path, "wb"))

    score = np.mean(list_inf)
    return -score

def compute_diversity(xai_sol:str, parameters:str, context:dict):
    #TODO add docstring
    prototypes = get_prototypes(xai_sol, parameters, context)
    distance = context['distance']
    scores = []
    #for each subset of data evaluate their prototypes
    for p in prototypes:
        dists = pairwise_distances(p,metric=distance)
        n = len(dists)
        scores.append(np.sum(dists)/(n**2-n))
    return np.mean(scores)

def compute_diversity_v2(prototypes,context):
    #TODO add docstring
    distance = context['distance']
    scores = []
    #for each subset of data evaluate their prototypes
    for p in prototypes:
        dists = pairwise_distances(p,metric=distance)
        n = len(dists)
        scores.append(np.sum(dists)/(n**2-n))
    return np.mean(scores)

def compute_non_representativeness(xai_sol:str, parameters:str, context:dict):
    #TODO add docstring
    prototypes = get_prototypes(xai_sol, parameters, context)
    distance = context['distance']
    X = context['X']
    scores = []
    for p in prototypes:
        scores.append(np.mean(np.min(pairwise_distances(X,p,metric=distance),axis=1)))
    return -np.mean(scores)

def compute_non_representativeness_v2(prototypes, context:dict):
    #TODO add docstring
    distance = context['distance']
    X = context['X']
    scores = []
    for p in prototypes:
        scores.append(np.mean(np.min(pairwise_distances(X,p,metric=distance),axis=1)))
    return -np.mean(scores)

def evaluate(xai_sol, parameters, property, context):
    """
    Evaluates an XAI solution on a given property.

    Parameters
    ----------
    xai_sol : str
        Name of the XAI solution that is evaluated.
    parameters : dict
        Parameters of the XAI solution for the current evaluation.
    property : str
        Property that is to evaluate with its corresponding evaluation measure.
    context : dict
        Information of the context that may change the process.

    Returns
    -------
    float
        Score for the evaluation measure corresponding to the property.
    """    
    # Set up of XAI solutions before computing evaluation
    if xai_sol in ['LIME','SHAP','Protodash']:
        context['explainer'] = set_up_explainer(xai_sol, parameters, context)

    #Computing explanations once for all evaluation metrics
    if xai_sol in ['MMD','kmedoids','Protodash']:
        if context['explanations'] == []:
            context['explanations'] = get_prototypes(xai_sol, parameters, context)
    
    # Computing evaluation for specified property
    if property == 'robustness':
        if context['question']=="Why":
            score = compute_lipschitz_robustness(xai_sol, parameters, context)
    if property == 'fidelity':
        if context['question']=="Why":
            score = compute_infidelity(xai_sol, parameters, context)
    if property == 'conciseness':
        if context['question']=="Why":
            score = -parameters['nfeatures']
        if context['question']=="What":
            score = -parameters['nb_proto']
    if property == 'diversity':
            if context['question']=="What":
                # score = compute_diversity(xai_sol,parameters,context)
                score = compute_diversity_v2(context['explanations'],context)
    if property == 'representativeness':
            if context['question']=="What":
                # score = compute_non_representativeness(xai_sol,parameters,context)
                score = compute_non_representativeness_v2(context['explanations'],context)
    
    return score

#TODO move it to utils or directly to launch
def linear_scalarization(score_hist, properties_list, context):
    """
    Aggregates the scores of the different evaluation measures by scaling and weighting them.

    Parameters
    ----------
    score_hist : dict
        History of scores on all evaluation measures.
    properties_list : list
        List of the evaluated properties.
    context : dict
        Information of the context that may change the process.

    Returns
    -------
    float
        Aggregated score.
    """    
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
            score_hist["aggregated_score"] += np.asarray(score_hist["scaled_"+property]).reshape(score_hist["aggregated_score"].shape) * weights[i]/len(properties_list)
        else :
            score_hist["scaled_"+property] = 0

    score_hist["aggregated_score"]=list(score_hist["aggregated_score"])
    context['explanations']=[]

    return score_hist