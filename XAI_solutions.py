import lime
import lime.lime_tabular
import numpy as np
import shap

from utils import reorder_attributes


def set_up_explainer(xai_sol, parameters, context):
    if xai_sol == "LIME":

        X=context["X"]
        y=context["y"]
        feature_names=context["feature_names"]
        verbose=context["verbose"]
        mode=context["task"]

        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X, feature_names=feature_names, training_labels=y, verbose=verbose, mode=mode, discretize_continuous=False)    
    
    elif xai_sol == "SHAP":
        #TODO use shap.Explainer for genericity
        X=context["X"]
        m = context['model']
        summarize = parameters['summarize']

        if summarize == "Sampling":
            explainer = shap.explainers.Sampling(m.predict, X)
        else :
            explainer = shap.KernelExplainer(m.predict, X)


    return explainer

def get_local_exp(xai_sol, x, parameters, context):
    if xai_sol == "LIME":

        explainer = context['explainer']
        m = context['model']
        feature_names = context['feature_names']
        num_samples = parameters['num_samples']

        e = reorder_attributes(dict(explainer.explain_instance(x, m.predict, num_samples=num_samples).as_list()), feature_names)

    if xai_sol == "SHAP":
        explainer = context['explainer']
        nsamples = parameters['nsamples']
        l1_reg = parameters['l1_reg']

        e = explainer.shap_values(x,nsamples=nsamples,l1_reg=l1_reg)

    return e[:parameters['nfeatures']]
