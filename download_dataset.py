import argparse

from sklearn.datasets import load_diabetes
import pandas as pd

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
    if name=="diabetes":
        diabetes = load_diabetes()

        X = diabetes.data
        y = diabetes.target
        feature_names = diabetes.feature_names

        df_X = pd.DataFrame(X,columns=feature_names)
        df_y = pd.Series(y,name='diabete progression')
        df_X["measure of disease progression"] = df_y
        df_X.to_csv("data/diabetes.csv",index=False)
    elif name=="pima indians":
        print("Go to https://www.kaggle.com/uciml/pima-indians-diabetes-database")
        print("Join the competition if necessary, download the dataset, extract it and copy it in the data/ folder.")
    else:
        raise ValueError("This dataset is not implemented yet.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'Download datasets for AutoXAI.')
    parser.add_argument('dataset', help='Dataset name, can be "diabetes", "pima indians"')

    args = parser.parse_args()
    dataset = args.dataset
    download_dataset(dataset)
