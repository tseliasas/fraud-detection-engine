import pandas as pd

def load_data(path):
    df = pd.read_csv(path)

    return df

def clean(df):
    df = 

if __name__ == "__main__":
    df = load_data("../data/raw/creditcard.csv")
    print(df.shape)
