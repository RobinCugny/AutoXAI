'''
MIT License

Copyright (c) 2022 Robin Cugny, IRIT and SolutionData Group, <robin.cugny@irit.fr>
Copyright (c) 2022 Julien Aligon, IRIT, <julien.aligon@irit.fr>
Copyright (c) 2022 Max Chevalier, IRIT, <max.chevalier@irit.fr>
Copyright (c) 2022 Geoffrey Roman Jimenez, SolutionData Group, <groman-jimenez@solutiondatagroup.fr>
Copyright (c) 2022 Olivier Teste, IRIT, <olivier.teste@irit.fr>

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

import argparse

from sklearn.datasets import load_diabetes
import pandas as pd

# TODO add UCI SPAM dataset
def download_dataset(name):
    """Downloads the corresponding dataset and formats it for AutoXAI

    Parameters
    ----------
    name : str
        The name of the dataset (ex: "diabetes", "pima indians")

    Raises
    ------
    ValueError
        If the dataset is not implemented yet.
    """
    if name == "diabetes":
        diabetes = load_diabetes()

        X = diabetes.data
        y = diabetes.target
        feature_names = diabetes.feature_names

        df_X = pd.DataFrame(X, columns=feature_names)
        df_y = pd.Series(y, name='diabete progression')
        df_X["measure of disease progression"] = df_y
        df_X.to_csv("data/diabetes.csv", index=False)
    # TODO redo with the API if possible maybe without kaggle
    elif name == "pima indians":
        print("Go to https://www.kaggle.com/uciml/pima-indians-diabetes-database")
        print("Join the competition if necessary, download the dataset, extract it and copy it in the data/ folder.")
    else:
        raise ValueError("This dataset is not implemented yet.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'Download datasets for AutoXAI.')
    parser.add_argument(
        'dataset', help='Dataset name, can be "diabetes", "pima indians"')

    args = parser.parse_args()
    dataset = args.dataset
    download_dataset(dataset)
