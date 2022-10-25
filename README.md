# AutoXAI
A framework for selecting and tuning the XAI (eXplainable AI) solutions you need.

## Acknowledgement
This framework has been proposed in [this paper](https://doi.org/10.1145/3511808.3557247) and also accessible [here](https://arxiv.org/abs/2210.02795 "AutoXAI: arXiv version") for the arXiv version. It is a collaboration between IRIT and SolutionData Group and the authors are Robin Cugny, Julien Aligon, Max Chevalier, Geoffrey Roman Jimenez and Olivier Teste.

Special thanks to Elodie Escriva for helping me with MMD-critic code and for her advice. Thanks to Vincent Fraysse for the help he gave me on the code, the scientific discussions and his unfailing support.

## Get datasets
To obtain the datasets of the paper, I am working on a `download_dataset.py` script but now the best way to obtain them is to access this [google drive link](https://drive.google.com/file/d/1mP7FrK9WSR8FCMCk02sLaK7KYt9fdm0t/view?usp=sharing). Download, uncompress and add them to the `data/` folder.

These datasets are modified version of the originals which are from:

1. Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani (2004) “Least Angle Regression,” Annals of Statistics (with discussion), 407-499. (https://web.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf)

2. Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. In Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261--265). IEEE Computer Society Press.

3. Almeida, T.A., Gomez Hidalgo, J.M., Yamakami, A. Contributions to the Study of SMS Spam Filtering: New Collection and Results. Proceedings of the 2011 ACM Symposium on Document Engineering (DOCENG'11), Mountain View, CA, USA, 2011.

Made accessible by https://archive.ics.uci.edu/ml/index.php

## To launch the experiments of the paper:
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
