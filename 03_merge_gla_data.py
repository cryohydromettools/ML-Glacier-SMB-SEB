import xarray as xr
from pathlib import Path
import pandas as pd

# Input directory
dir_data = Path('/home/lacrio/DATA/CB/pros')

# Glacier names and IDs
gla_names = ['Shallap', 'Ghueshgue', 'Yanamarey', 'Artesonraju']
gla_ids   = ['G1', 'G2', 'G3', 'G4']



dfs = []

for gla, gid in zip(gla_names, gla_ids):
    int_file = dir_data / f"{gla}_cpy_era5_l.nc"
    
    with xr.open_dataset(int_file) as ds:
        df = ds.to_dataframe().reset_index()
        df["GLACIER"] = gid        # ← add glacier ID
        dfs.append(df)

# Concatenate all glaciers into a single DataFrame
df_all = pd.concat(dfs, ignore_index=True).dropna()
print(df_all)
# Output file
out_filename = dir_data / f"data_train_test_glas.csv"

# Save to CSV
df_all.to_csv(out_filename, index=False)
