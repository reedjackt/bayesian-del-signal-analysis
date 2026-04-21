import pandas as pd
import numpy as np

# Create 10,000 rows of fake KinDEL data
data = {
    "molecule_hash": [f"hash_{i}" for i in range(10000)],
    "smiles": ["C" * (i % 10 + 1) for i in range(10000)],
    "pre-selection_counts": np.random.poisson(20, 10000),
    "target_replicate_1": np.random.poisson(5, 10000),
    "target_replicate_2": np.random.poisson(5, 10000),
    "target_replicate_3": np.random.poisson(25, 10000),  # Replicate 3 is "deeper"
}

# Add a few "Hits"
for i in range(10):
    data["target_replicate_1"][i] = 200
    data["target_replicate_2"][i] = 180
    data["target_replicate_3"][i] = 1000

df = pd.DataFrame(data)
df.to_csv("data/kindel_ddr1.csv", index=False)
print("Successfully created data/kindel_ddr1.csv with KinDEL schema.")

