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

import numpy as np
from numpy.random import randint, choice, rand, uniform
from utils import hp_possible_values
from bayes_opt import BayesianOptimization
from XAI_solutions import set_up_explainer, get_local_exp
from evaluation_measures import evaluate,linear_scalarization

def get_parameters(xai_sol, score_hist, hpo, properties_list, context):
    """
    Generates the hyparameters for an XAI solution according to the chosen HPO method.

    Parameters
    ----------
    xai_sol : str
        Name of the XAI solution for which we want parameters to be generated.
    score_hist : dict
        History of scores on all evaluation measures.
    hpo : str
        Method to use to generate hyperparameters.
    properties_list : list
         List of the evaluated properties (conciseness impacts hyperparameters).
    context : dict
        Information of the context that may change the process.

    Returns
    -------
    dict
        Parameters of the XAI solution for the next explanations and evaluations.
    """    
    parameters = {}

    if hpo == 'random':
        if xai_sol == 'LIME':
            parameters['num_samples'] = randint(hp_possible_values["LIME"]["num_samples"][0],
                                                hp_possible_values["LIME"]["num_samples"][1])
        if xai_sol == 'SHAP':
            parameters['summarize'] = choice(hp_possible_values["SHAP"]["summarize"])
            parameters['nsamples'] = randint(hp_possible_values["SHAP"]["nsamples"][0],
                                             hp_possible_values["SHAP"]["nsamples"][1])
            parameters['l1_reg'] = choice(hp_possible_values["SHAP"]["l1_reg"])
            if parameters['l1_reg'] == "float":
                parameters['l1_reg'] = rand()
            if parameters['l1_reg'] == 'num_features(int)':
                parameters['l1_reg'] = 'num_features('+str(randint(1,context["X"].shape[1]))+')'

        if xai_sol == 'MMD':
            parameters['gamma'] = uniform(hp_possible_values["MMD"]["gamma"][0],hp_possible_values["MMD"]["gamma"][1])
            # parameters['ktype'] = choice(hp_possible_values["MMD"]["ktype"])
        if xai_sol == 'Protodash':
            parameters['kernelType'] = choice(hp_possible_values["Protodash"]["kernelType"])
            parameters['sigma'] = uniform(hp_possible_values["Protodash"]["sigma"][0],hp_possible_values["Protodash"]["sigma"][1])
        if xai_sol == 'kmedoids':
            parameters['metric'] = choice(hp_possible_values["kmedoids"]["metric"])
            parameters['method'] = choice(hp_possible_values["kmedoids"]["method"])
            parameters['init'] = choice(hp_possible_values["kmedoids"]["init"])
            parameters['max_iter'] = randint(hp_possible_values["kmedoids"]["max_iter"][0],
                                             hp_possible_values["kmedoids"]["max_iter"][1])
        parameters['nb_proto'] = randint(2,min(np.unique(context["y"],return_counts=True)[1]))
        if 'conciseness' in properties_list:
            parameters['nfeatures'] = randint(1,len(context["feature_names"]))
        else:
            parameters['nfeatures'] = len(context["feature_names"])

    if hpo == "default":
        if xai_sol == 'LIME':
            parameters['num_samples'] = 5000
        if xai_sol == 'SHAP':
            parameters['summarize'] = "KernelExplainer"
            parameters['nsamples'] = 2048
            parameters['l1_reg'] = "auto"

        if context["question"]=="Why":
            parameters['nfeatures'] = len(context["feature_names"])
        if context["question"]=="What":
            parameters['nb_proto'] = 3
            
        if xai_sol == 'MMD':
            parameters['gamma'] = None
        if xai_sol == 'Protodash':
            parameters['kernelType'] = 'other'
            parameters['sigma'] = 2
        if xai_sol == 'kmedoids':
            parameters['metric'] = "euclidean"
            parameters['method'] = "alternate"
            parameters['init'] = "build"
            parameters['max_iter'] = 300

        # parameters['most_influent_features'] = list(np.arange(parameters['nfeatures']))
    return parameters

def gp_optimization(xai_sol, score_hist, properties_list, context, epochs):
    """
    Generates the hyparameters for an XAI solution using Gaussian Process method.

    Parameters
    ----------
    xai_sol : str
        Name of the XAI solution for which we want parameters to be generated.
    score_hist : dict
        History of scores on all evaluation measures.
    properties_list : list
        List of the evaluated properties (conciseness impacts hyperparameters).
    context : dict
        Information of the context that may change the process.
    epochs : int
        Number of iterations in the gaussian process.

    Returns
    -------
    list
        List of dictionaries, each of them contains parameters.
    """
    #TODO use utils and hp_possible_values
    #TODO end of the f function is the same, create a function for it
    pbounds = {}
    if xai_sol=='LIME':
        pbounds = {'num_samples': (10, 10000), 'nfeatures':(1,len(context["feature_names"]))}#use utils
        init_points = 5**len(pbounds)

        def f(num_samples,nfeatures):
            parameters = {'num_samples':int(num_samples),'nfeatures':int(np.round(nfeatures))}

            for property in properties_list:
                property_score = evaluate(xai_sol, parameters, property, context)
                score_hist[property].append(property_score)
            linear_scalarization(score_hist, properties_list, context)
            score = score_hist["aggregated_score"][-1]
            return score

    if xai_sol=='SHAP':
        pbounds = {'summarize':(0,1),'nsamples': (10, 2048), 'l1_reg':(0,3), 'num_features':(1,len(context["feature_names"])), 'nfeatures':(1,len(context["feature_names"]))}
        init_points = 5**2
        #num_features is for l1_reg and nfeatures for size of explanation vector
        def f(summarize, nsamples, l1_reg, num_features, nfeatures):
            parameters = {}
            parameters['nsamples'] = int(nsamples)
            parameters['summarize'] = hp_possible_values["SHAP"]["summarize"][int(np.round(summarize))]
            parameters['l1_reg'] = hp_possible_values["SHAP"]["l1_reg"][int(np.round(l1_reg))]
            parameters['nfeatures'] = int(np.round(nfeatures))
            if parameters['l1_reg'] == 'num_features(int)':
                parameters['l1_reg'] = 'num_features('+str(int(np.round(num_features)))+')'

            for property in properties_list:
                property_score = evaluate(xai_sol, parameters, property, context)
                score_hist[property].append(property_score)
            linear_scalarization(score_hist, properties_list, context)
            score = score_hist["aggregated_score"][-1]
            return score

    if xai_sol == 'MMD':
        pbounds = {'nb_proto': (2,min(np.unique(context["y"],return_counts=True)[1])), 'gamma':(0,1)}
        # init_points = 3**2

        def f(nb_proto,gamma):
            parameters = {'nb_proto':int(nb_proto),'gamma':gamma}
            for property in properties_list:
                property_score = evaluate(xai_sol, parameters, property, context)
                score_hist[property].append(property_score)
            linear_scalarization(score_hist, properties_list, context)
            score = score_hist["aggregated_score"][-1]
            return score

    if xai_sol == 'Protodash':
        #parameters['kernelType'] = choice(hp_possible_values["Protodash"]["kernelType"])
        pbounds = {'nb_proto': (2,min(np.unique(context["y"],return_counts=True)[1])-2),
                    'sigma':(0,30),
                    'kernelType':(0,1)}
        # init_points = 3**2

        def f(nb_proto,sigma,kernelType):
            parameters = {'nb_proto':int(nb_proto),'sigma':sigma}
            parameters['kernelType'] = hp_possible_values["Protodash"]["kernelType"][int(np.round(kernelType))]
            # parameters['kernelType'] = 'other'
            for property in properties_list:
                property_score = evaluate(xai_sol, parameters, property, context)
                score_hist[property].append(property_score)
            linear_scalarization(score_hist, properties_list, context)
            score = score_hist["aggregated_score"][-1]
            return score

    if xai_sol == 'kmedoids':
        pbounds = {'nb_proto': (2,min(np.unique(context["y"],return_counts=True)[1])), 
                    'metric':(0,2),
                    'method':(0,1),
                    'init':(0,3),
                    'max_iter':(0,300)}
        # init_points = 3**2

        def f(nb_proto, metric, method, init, max_iter):
            parameters = {}
            parameters['nb_proto'] = int(nb_proto)
            parameters['metric'] = hp_possible_values["kmedoids"]["metric"][int(np.round(metric))]
            parameters['method'] = hp_possible_values["kmedoids"]["method"][int(np.round(method))]
            parameters['init'] = hp_possible_values["kmedoids"]["init"][int(np.round(init))]
            parameters['max_iter'] = int(np.round(max_iter))
            for property in properties_list:
                property_score = evaluate(xai_sol, parameters, property, context)
                score_hist[property].append(property_score)
            linear_scalarization(score_hist, properties_list, context)
            score = score_hist["aggregated_score"][-1]
            return score
    
    # init_points = 3*len(pbounds)#TODO find better init (square is better but expensive)
    init_points = 3**2
    optimizer = BayesianOptimization(
        f=f,
        pbounds=pbounds,
        verbose=0,
        random_state=1,
    )
    
    optimizer.maximize(
        init_points=init_points,
        n_iter=epochs,
    )

    return optimizer.res