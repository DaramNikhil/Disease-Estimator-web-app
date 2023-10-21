import pandas as pd
# sample systamatic file
data_path = "code/artifacts/Test_data.csv"

with open(data_path) as f:
    features = pd.read_csv(f)
    features = features.reset_index(drop=True)

my_val = features["Disease"]
my_val = [my_val]
print(my_val)
