import pandas as pd
import sys

file_name = sys.argv[1]

def top_x(df, x=3):
    df = df.sort_values("avg_score", ascending=False)
    return df.head(x)

def bottom_x(df, x=3):
    df = df.sort_values("avg_score")
    return df.head(x)

if __name__ == "__main__":
    # read csv
    results_df = pd.read_csv(file_name)

    # extract useful info
    print(top_x(results_df, 25))
