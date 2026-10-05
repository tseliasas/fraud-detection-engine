import pandas as pd

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

if __name__ == "__main__":
    df = load_data("../data/raw/creditcard.csv")
    print(df.shape)

    df = clean(df)
    print(df.shape)
    print(df["hour"].min(), df["hour"].max())
    print(df["Class"].value_counts())
