import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)

    return df

SECONDS_PER_DAY = 86400
SECONDS_PER_HOUR = 3600

def clean(df):
    # Drop exact copies so they can't leak across the train/test split
    df = df.drop_duplicates()

    # Time is seconds since the first transaction (~midnight), so wrap to hour of day (0-23)
    df = df.assign(hour=(df["Time"] % SECONDS_PER_DAY) // SECONDS_PER_HOUR)
    df = df.drop(columns=["Time"])

    return df

def split(df, test_size, seed):
    X = df.drop(columns="Class")
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_data("../data/raw/creditcard.csv")
    print(df.shape)

    df = clean(df)
    print(df.shape)
    print(df["hour"].min(), df["hour"].max())
    print(df["Class"].value_counts())
    X_train, X_test, y_train, y_test = split(df, 0.2, 42)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.value_counts(normalize=True))
    print(y_test.value_counts(normalize=True))
