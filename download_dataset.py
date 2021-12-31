import argparse

from sklearn.datasets import load_diabetes
import pandas as pd

def download_dataset(name):
    if name=="diabetes":
        diabetes = load_diabetes()

        X = diabetes.data
        y = diabetes.target
        feature_names = diabetes.feature_names

        df_X = pd.DataFrame(X,columns=diabetes.feature_names)
        df_y = pd.Series(y,name='diabete progression')
        df_X["measure of disease progression"] = df_y
        df_X.to_csv("data/diabetes.csv",index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'Download datasets for AutoXAI.')
    parser.add_argument('dataset', help='Dataset name, can be "diabetes", "pima indians"')

    args = parser.parse_args()
    dataset = args.dataset
    download_dataset(dataset)
