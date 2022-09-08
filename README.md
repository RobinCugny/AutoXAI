# AutoXAI
A framework to automatically select the XAI solution you need

## Acknowledgement
This framework has been proposed in 'link here'. It is a collaboration between IRIT and SolutionData Group and the authors are Robin Cugny, Julien Aligon, Max Chevalier, Geoffrey Roman Jimenez and Olivier Teste.

## Launch experiment of the paper:
```
python launch.py data/diabetes.csv "measure of disease progression" regression -m models/MLPRegressor_diabetes.p -q "Why" -p "robustness,fidelity,conciseness" --hpo "gp" --evstrat "ES,IS" -w "1,2,0.5" --scaling "Std" -s 0 -e 25
```
or
```
python launch.py data/pima_indians_diabetes.csv "Outcome" classification -m models/MLPClassifier_pima_indians.p -q "Why" -p "robustness,fidelity,conciseness" --hpo "gp" --evstrat "ES,IS" -w "1,2,0.5" --scaling "Std" -s 0 -e 25
```
or
```
python launch.py data/SPAM_vec.csv "class" classification -q "What" -p "diversity,representativeness,conciseness" --hpo "gp" -w "1,2,2" -s 0 -e 25
```

## Help:
```
usage: Launch AutoXAI framework to find the XAI solution that maximize specified properties. [-h] [-m MODEL] [-q QUESTION] [-x XAI] [-e EPOCHS]
                                                                                             [-t TRIALS] [-p PROPERTIES] [--hpo HPO]
                                                                                             [--evstrat EVSTRAT] [--verbose] [-s SEED]
                                                                                             dataset label task

positional arguments:
  dataset               Path to the dataset that is used for the model training.
  label                 Name of the label column.
  task                  Type of task 'regression' or 'classification'.

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Path to the model that is to explain.
  -q QUESTION, --question QUESTION
                        Question that the XAI solution should answer, see documentation for the full list of the questions.
  -x XAI, --xai XAI     List of XAI solution to test ex : "LIME,SHAP", must answer to the same question, see documentation.
  -e EPOCHS, --epochs EPOCHS
                        Maximum mumber of epochs to optimize hyperparameters for each XAI solution.
  -t TRIALS, --trials TRIALS
                        Maximum number of XAI solution to evaluate.
  -p PROPERTIES, --properties PROPERTIES
                        List of properties to consider, ex : "robustness,fidelity,conciseness", must work on the set of XAI solutions, see
                        documentation.
  --hpo HPO             Hyperparameters optimization method, ex : "gp", see documentation for the full list of the hpo methods.
  --evstrat EVSTRAT     Evaluation strategies to use in order gain time, ex : "ES,IS", see documentation for the full list of the evaluation        
                        strategies.
  --verbose             Launch AutoXAI with verbosity on True.
  -s SEED, --seed SEED  Set up a seed for random generated numbers.
```


For any question concerning the paper or the framework, please contact me at robin.cugny@irit.fr