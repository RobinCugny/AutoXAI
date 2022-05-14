import sys
import os
import lime
import lime.lime_tabular
import numpy as np
import shap
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from aix360.algorithms.protodash import ProtodashExplainer
from mmdcritic import mmd_critic
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import OneHotEncoder
from utils import reorder_attributes


def set_up_explainer(xai_sol, parameters, context):
    """
    Initialize the explainer object for solutions that need it.

    Parameters
    ----------
    xai_sol : str
        Name of the XAI solution that is initialized.
    parameters : dict
        Parameters of the XAI solution for the initialization.
    context : dict
        Information of the context that may change the process.

    Returns
    -------
    object
        Explainer object that can be used to generate explanation.
    """    
    if xai_sol == "LIME":

        X=context["X"]
        y=context["y"]
        feature_names=context["feature_names"]
        verbose=context["verbose"]
        mode=context["task"]
        if mode == 'regression':
            explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X, feature_names=feature_names, training_labels=y, verbose=verbose, mode=mode, discretize_continuous=False)    
        else:
            class_names = np.unique(y)
            explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X, feature_names=feature_names, class_names=class_names, verbose=verbose, mode=mode, discretize_continuous=False)    

    
    elif xai_sol == "SHAP":
        #TODO use shap.Explainer for genericity
        X=context["X"]
        m = context['model']
        summarize = parameters['summarize']
        mode=context["task"]

        if mode=='regression':
            if summarize == "Sampling":
                explainer = shap.explainers.Sampling(m.predict, X)
            else :
                explainer = shap.KernelExplainer(m.predict, X)
        else:
            if summarize == "Sampling":
                explainer = shap.explainers.Sampling(m.predict_proba, X)
            else :
                explainer = shap.KernelExplainer(m.predict_proba, X)
    
    elif xai_sol=='Protodash':
        explainer = ProtodashExplainer()
    else:
        return 0
    return explainer

def get_local_exp(xai_sol, x, parameters, context, update_order_feat=True):
    """
    Calculates a local explanation and formats it for future evaluation.

    Parameters
    ----------
    xai_sol : str
        Name of the XAI solution that is used to get explanation.
    x : list or numpy array
        Data point for which we want the explanation of the black box 
    parameters : dict
        Parameters of the XAI solution for the local explanation.
    context : dict
        Information of the context that may change the process.

    Returns
    -------
    list
        Vector of feature influence constituting the explanation.
    """    
    explainer = context['explainer']
    m = context['model']
    mode=context["task"]
    if xai_sol == "LIME":
        num_samples = parameters['num_samples']
        feature_names = context['feature_names']

        if mode == 'regression':
            e = reorder_attributes(dict(explainer.explain_instance(x, m.predict, num_samples=num_samples).as_list()), feature_names)
        else:
            # label = int(m.predict(x.reshape(1, -1)))
            # print(explainer.explain_instance(x, m.predict_proba, num_samples=num_samples).as_list())
            e = reorder_attributes(dict(explainer.explain_instance(x, m.predict_proba, num_samples=num_samples).as_list()), feature_names)

    if xai_sol == "SHAP":
        nsamples = parameters['nsamples']
        l1_reg = parameters['l1_reg']
        if mode == 'regression':
            e = explainer.shap_values(x,nsamples=nsamples,l1_reg=l1_reg)
        else:
            pred=int(m.predict(x.reshape(1, -1)))
            e = explainer.shap_values(x,nsamples=nsamples,l1_reg=l1_reg)[pred]
        # print("------SHAP-----")
        # print(x)
        # print(e)
    if update_order_feat:
        most_influent_features = np.argsort(np.abs(e))[::-1][:parameters['nfeatures']]
        parameters['most_influent_features'] = most_influent_features
    e=list(np.asarray(e)[parameters['most_influent_features']])
    return e

def get_prototypes(xai_sol, parameters, context):
    #TODO add docstring
    X=context['X']
    y=context['y']
    nb_proto = parameters['nb_proto']
    prototypes=[]
    if xai_sol == "Protodash":
        kernelType=parameters['kernelType']
        sigma=parameters['sigma']
        explainer = context['explainer']
        for c in np.unique(y):
            subset = X[np.where(y==c)]
            # print(subset.shape,nb_proto,kernelType,sigma)
            with open(os.devnull, "w") as devnull:
                old_stdout = sys.stdout
                sys.stdout = devnull
                try:  
                    (W, S, _) = explainer.explain(subset, subset, m=nb_proto,kernelType=kernelType,sigma=sigma)
                finally:
                    sys.stdout = old_stdout
            prototypes.append(subset[S])
    if xai_sol == "MMD":
        gamma = parameters['gamma']
        # ktype = parameters['ktype']
        for c in np.unique(y):
            nb_critic = 1
            subset = X[np.where(y==c)]
            subset_y = y[np.where(y==c)]
            prototypes_idx = mmd_critic(subset, subset_y, nb_proto, nb_critic, gamma=gamma, ktype=0, crit=False)
            prototypes.append(subset[prototypes_idx])
    if xai_sol == "kmedoids":
        metric=parameters['metric']
        method=parameters['method']
        init=parameters['init']
        max_iter=parameters['max_iter']
        for c in np.unique(y):
            subset = X[np.where(y==c)]
            kmedoids = KMedoids(n_clusters=nb_proto, metric=metric, method=method, init=init, max_iter=max_iter, random_state=0).fit(subset)
            medoids_idx = kmedoids.medoid_indices_
            prototypes.append(subset[medoids_idx])

    return prototypes

def get_exp_std(xai_sol, parameters, context):
    #TODO complete docstring
    """ Compute the standard deviation of explanations to scale them for fidelity computation

    Parameters
    ----------
    xai_sol : _type_
        _description_
    parameters : _type_
        _description_
    context : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """    
    n=10
    X=context["X"]
    exp_values = []
    context['explainer'] = set_up_explainer(xai_sol, parameters, context)
    # print(get_local_exp(xai_sol, X[0], parameters, context))
    for i in range(n):
        exp_values += get_local_exp(xai_sol, X[i], parameters, context)
    
    std = np.std(exp_values)
    context[xai_sol+"_std"]=std
    return std
