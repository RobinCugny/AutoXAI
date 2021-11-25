import numpy as np
from numpy.random import randint, choice, rand
from utils import hp_possible_values
from bayes_opt import BayesianOptimization
from XAI_solutions import set_up_explainer, get_local_exp
from evaluation_measures import evaluate,linear_scalarization

def get_parameters(xai_sol, score_hist, hpo, context):
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

    if hpo == "default":
        if xai_sol == 'LIME':
            parameters['num_samples'] = 5000
        if xai_sol == 'SHAP':
            parameters['summarize'] = "KernelExplainer"
            parameters['nsamples'] = 2048
            parameters['l1_reg'] = "auto"
    return parameters

def gp_optimization(xai_sol, score_hist, properties_list, context, epochs, weights):
    pbounds = {}
    init_points = 5**len(pbounds)
    if xai_sol=='LIME':
        pbounds = {'num_samples': (10, 10000)}#use utils

        def f(num_samples):
            parameters = {'num_samples':int(num_samples)}

            for property in properties_list:
                property_score = evaluate(xai_sol, parameters, property, context)
                score_hist[property].append(property_score)
            linear_scalarization(score_hist, properties_list, context)
            score = score_hist["aggregated_score"][-1]
            return score

    if xai_sol=='SHAP':
        pbounds = {'summarize':(0,1),'nsamples': (10, 2048), 'l1_reg':(0,3), 'num_features':(1,context["X"].shape[1])}

        def f(summarize, nsamples, l1_reg, num_features):
            parameters = {}
            parameters['nsamples'] = int(nsamples)
            parameters['summarize'] = hp_possible_values["SHAP"]["summarize"][int(np.round(summarize))]
            parameters['l1_reg'] = hp_possible_values["SHAP"]["l1_reg"][int(np.round(l1_reg))]
            if parameters['l1_reg'] == 'num_features(int)':
                parameters['l1_reg'] = 'num_features('+str(num_features)+')'

            for property in properties_list:
                property_score = evaluate(xai_sol, parameters, property, context)
                score_hist[property].append(property_score)
            linear_scalarization(score_hist, properties_list, context)
            score = score_hist["aggregated_score"][-1]
            return score
    
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