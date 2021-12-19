import argparse
from utils import load_dataset, load_model, questions_to_xai_sol, hpo_list
from evaluation_measures import evaluate,linear_scalarization
from hyperparameters_optimization import get_parameters, gp_optimization
from XAI_solutions import get_exp_std
import numpy as np
from time import time

#TODO rename file in core (according to paper)

def main(dataset_path, label, task, model_path=None, question=None, xai_list=None, epochs=10, trials=None, properties_list=None, hpo=None, evstrat_list=None, verbose=False, seed=None, weights=[-1,-2,-0.5], scaling="Std", session_id = '0'):
    
    # weights=[-1,-1]
    # weights=[-1,-2,-0.5] #TODO set these as parameters
    # # scaling="MinMax"
    # scaling="Std"
    # session_id = '0'
    evstrat_list = ['ES','IS']
    if 'ES' in evstrat_list:
        early_stopping = True
    if 'IS' in evstrat_list:
        information_sharing = True

    start_time = time()

    if seed!=None:
        np.random.seed(seed)
    """
    Check parameters section
    """

    if question == None:
        if xai_list == None:
            raise ValueError("A question or a set of XAI solutions is necessary")
        else:
            #TODO check if all elements of xai_list are from the same set of xai solutions, infer the question
            for q, xai_in_q in questions_to_xai_sol.items():
                if set(xai_list).issubset(xai_in_q):
                    question = q
                    break
    else:
        if question not in questions_to_xai_sol.keys():
            raise NameError("The question", question ,"is not in",questions_to_xai_sol.keys())

        if xai_list == None:
            xai_list = questions_to_xai_sol[question]
            print("XAI solutions evaluated",xai_list)
        else:
            for xai in xai_list:
                if xai not in questions_to_xai_sol[question]:
                    raise ValueError("Some XAI solutions are not in the same set or are not implemented yet")

    #TODO convert properties in evaluation measures

    X,y,feature_names = load_dataset(dataset_path, label)

    context = {}

    context["X"]=X
    context["y"]=y
    context["feature_names"] = feature_names
    context["verbose"] = verbose
    context["task"] = task
    context["question"] = question
    context["scaling"] = scaling
    context["weights"] = weights

    if question == "Why":
        if model_path == None:
            raise ValueError("Model is necessary for this question")
        context['model'] = load_model(model_path)

        score_hist = {}
        score_hist["xai_sol"] = []
        score_hist["epoch"] = []
        score_hist["aggregated_score"] = []
        score_hist["parameters"] = []

        for property in properties_list:
            score_hist[property]=[]
            score_hist["scaled_"+property]=[]

        # Evaluate each xai_sol with default parameters
        print("Evaluate each XAI solution with its default parameters.")
        for xai_sol in xai_list:
            print("XAI solution:",xai_sol)
            score_hist["xai_sol"].append(xai_sol)
            score_hist["epoch"].append(-1)

            parameters = get_parameters(xai_sol, score_hist, "default", properties_list, context)
            score_hist["parameters"].append(parameters)
            get_exp_std(xai_sol, parameters, context)
            print("  parameters:",parameters)
            for property in properties_list:
                score = evaluate(xai_sol, parameters, property, context)
                score_hist[property].append(score)
                print("    "+property+" loss:",score)
            linear_scalarization(score_hist, properties_list, context) #TODO remove ? This does note make any sens at this point
            
        # TODO keep the best performing (nb = trials)
        if trials != None:
            pass

        for xai_sol in xai_list:
            print("XAI solution:",xai_sol)
            if hpo == "random":
                for e in range(epochs) :
                    score_hist["xai_sol"].append(xai_sol)
                    score_hist["epoch"].append(e)
                    print("  epoch:",e)

                    parameters = get_parameters(xai_sol, score_hist, hpo, properties_list, context)
                    score_hist["parameters"].append(parameters)
                    print("  parameters:",parameters)

                    for property in properties_list:
                        score = evaluate(xai_sol, parameters, property, context)
                        print("    "+property+" loss:",score)
                        score_hist[property].append(score)

                    linear_scalarization(score_hist, properties_list, context)
                    # print("ES tot")
                    # print(len(score_hist['aggregated_score']) - np.argmax(score_hist['aggregated_score']))
                    if early_stopping and len(score_hist['aggregated_score']) - np.argmax(score_hist['aggregated_score']) > 5:
                        break
                    # print("aggregated_score",score_hist["aggregated_score"])
                    
            elif hpo == "gp":
                results = gp_optimization(xai_sol, score_hist, properties_list, context, epochs)
                # print(results)
                for e, r in enumerate(results):
                    score_hist["xai_sol"].append(xai_sol)
                    score_hist["epoch"].append(e)
                    score_hist["parameters"].append(r['params'])

            else:
                raise NameError("The optimization", hpo,"is not in", hpo_list)
    
    # Display best performing XAI solution with details
    best_performing_solution = np.argmax(score_hist['aggregated_score'])
    print("\n---------------------- Best performing XAI solution ----------------------")
    print("XAI solution:",score_hist['xai_sol'][best_performing_solution])
    print("parameters:",score_hist['parameters'][best_performing_solution])
    print('\nscores:')
    for property in properties_list:
        print("  ",property,"loss:", score_hist[property][best_performing_solution])
    
    print("\n----------------------------- Score details -----------------------------")
    print('scaled scores:')
    for property in properties_list:
        print("  ","scaled "+property+":", score_hist["scaled_"+property][best_performing_solution])
    print("weights:",weights)
    print("aggregated score:",score_hist['aggregated_score'][best_performing_solution])

    print("\n------------------------------ All records ------------------------------")
    for k,v in score_hist.items():
        print(k+":")
        print("  ",v)

    with open('results/best_sol_'+session_id+'.txt', 'w') as f:
        # f.write("XAI solution:\n")
        # f.write(str(score_hist['xai_sol'][best_performing_solution]))
        # f.write("\nparameters:\n")
        # f.write(str(score_hist['parameters'][best_performing_solution])+"\n")
        
        for i in np.argsort(score_hist['aggregated_score'])[::-1]:
            f.write('\n\nxai sol: '+str(score_hist['xai_sol'][i]))
            f.write('\nparams: '+str(score_hist['parameters'][i]))
            f.write('\nag scores: '+str(score_hist['aggregated_score'][i]))
            for property in properties_list:
                f.write('\n'+property+': '+str(score_hist[property][i])+' scaled:'+str(score_hist['scaled_'+property][i]))
        f.write("\n--------------------RECORDS--------------------")
        for k,v in score_hist.items():
            f.write("\n"+k+":")
            f.write("\n  "+str(v))

    print(time()-start_time,"sec elapsed")

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'Launch AutoXAI framework to find the XAI solution that maximize specified properties.')
    parser.add_argument('dataset', help='Path to the dataset that is used for the model training.')
    parser.add_argument('label', help='Name of the label column.')
    parser.add_argument('task', help="Type of task 'regression' or 'classification'.")
    parser.add_argument('-m', '--model', help='Path to the model that is to explain.')
    parser.add_argument('-q', '--question', help='Question that the XAI solution should answer, see documentation for the full list of the questions.')
    parser.add_argument('-x', '--xai', help='List of XAI solution to test ex : "LIME,SHAP", must answer to the same question, see documentation.')
    parser.add_argument('-e', '--epochs', type=int, help='Maximum mumber of epochs to optimize hyperparameters for each XAI solution.', default=10)
    parser.add_argument('-t', '--trials', help='Maximum number of XAI solution to evaluate.')
    parser.add_argument('-p', '--properties', help='List of properties to consider, ex : "robustness,fidelity,conciseness", must work on the set of XAI solutions, see documentation.')
    parser.add_argument('--hpo', help='Hyperparameters optimization method, ex : "gp", see documentation for the full list of the hpo methods.')
    parser.add_argument('--evstrat', help='Evaluation strategies to use in order gain time, ex : "ES,IS", see documentation for the full list of the evaluation strategies.')
    parser.add_argument('--verbose', help='Launch AutoXAI with verbosity on True.', action="store_true")
    parser.add_argument('-s','--seed', type=int, help='Set up a seed for random generated numbers.')

    args = parser.parse_args()

    dataset_path = args.dataset
    label = args.label
    task = args.task
    model_path = args.model
    question = args.question
    xai_list = str(args.xai).split(",")
    epochs = args.epochs
    trials = args.trials
    properties_list = str(args.properties).split(",")
    hpo = args.hpo
    evstrat_list = str(args.evstrat).split(",")
    verbose = args.verbose
    seed = args.seed

    if xai_list == ['None']:
        xai_list = None
    if properties_list == ['None']:
        properties_list = None
    if evstrat_list == ['None']:
        evstrat_list = None

    main(dataset_path, label, task, model_path, question, xai_list, epochs, trials, properties_list, hpo, evstrat_list, verbose, seed)
