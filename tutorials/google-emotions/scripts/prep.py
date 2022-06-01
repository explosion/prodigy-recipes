import pathlib 
import pandas as pd
from collections import Counter


paths = pathlib.Path("data", "full_dataset").glob("*.csv")
df = pd.concat([pd.read_csv(path) for path in paths])

print(f"There are {len(df)} annotations and {df['id'].nunique()} unique examples.")
print(f"The excitement column has the following distribution: {Counter(df['excitement'])}")

subset = df[['text', 'id', 'rater_id', 'example_very_unclear', 'excitement']]
subset.to_csv("data/annotations.csv", index=False)

print("Subset available at data/annotations.csv")
