"""
Process COSIPY outputs:
- Concatenate per variable (not all at once)
- Compute daily sum or mean
- Merge final daily variables
"""

from pathlib import Path
import xarray as xr

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
dir_data     = Path('/home/lacrio/DATA/CB')
dir_data_out = Path('/home/lacrio/DATA/CB/pros')

# Glacier name (options: Artesonraju, Ghueshgue, Shallap, Yanamarey)
gla_name = 'Shallap'

var_sum  = ['MB', 'SNOWFALL', 'RAIN', 'surfM', 'subM',
            'SUBLIMATION', 'EVAPORATION', 'CONDENSATION',
            'DEPOSITION', 'REFREEZE', 'Q']

var_mean = ['ME', 'G', 'ALBEDO', 'LWin', 'LWout',
            'LE', 'H', 'B', 'QRR', 'TS']

sel_vars = ['MASK', 'HGT', 'SLOPE', 'ASPECT']

data_path = dir_data / gla_name
nc_files = sorted(data_path.glob('*.nc'))

# -----------------------------------------------------------------------------
# Static variables (only once)
# -----------------------------------------------------------------------------
ds_geo = xr.open_dataset(nc_files[0])[sel_vars]
MASK = ds_geo['MASK']

# -----------------------------------------------------------------------------
# Function to process one variable at a time
# -----------------------------------------------------------------------------
def process_variable(varname, method):
    
    print(f"\nProcessing variable: {varname}")
    
    var_list = []
    
    for f in nc_files:
        ds = xr.open_dataset(f)[varname]
        var_list.append(ds)
    
    # Concatenate only this variable
    ds_var = xr.concat(var_list, dim="time")
    
    # Sort and remove duplicates
    ds_var = ds_var.sortby("time")
    
    # Daily aggregation
    if method == "sum":
        ds_daily = ds_var.resample(time="1MS").sum()
    elif method == "mean":
        ds_daily = ds_var.resample(time="1MS").mean()
    
    # Apply mask
    ds_daily = ds_daily.where(MASK == 1)
    
    return ds_daily

# -----------------------------------------------------------------------------
# Process sum variables
# -----------------------------------------------------------------------------
daily_vars = []

for var in var_sum:
    daily_vars.append(process_variable(var, "sum"))

# -----------------------------------------------------------------------------
# Process mean variables
# -----------------------------------------------------------------------------
for var in var_mean:
    daily_vars.append(process_variable(var, "mean"))

# -----------------------------------------------------------------------------
# Merge all daily variables
# -----------------------------------------------------------------------------
ds_daily_all = xr.merge(daily_vars)

# -----------------------------------------------------------------------------
# Merge static + dynamic
# -----------------------------------------------------------------------------
ds_merged = xr.merge([ds_geo, ds_daily_all])

# -----------------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------------
out_file = dir_data_out / f"{gla_name}_monthly.nc"
ds_merged.to_netcdf(out_file)

print(f"\nSaved merged dataset: {out_file}")

