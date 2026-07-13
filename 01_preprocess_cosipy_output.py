"""
Process COSIPY outputs:
1) Read surface mass balance (MB) from multiple NetCDF files
2) Aggregate MB to monthly sums
3) Extract static topographic variables
4) Merge dynamic and static datasets
5) Save merged NetCDF file
"""

from pathlib import Path
import xarray as xr

# -----------------------------------------------------------------------------
# Paths and configuration
# -----------------------------------------------------------------------------
dir_data     = Path('/home/lacrio/DATA/CB')
dir_data_out = Path('/home/lacrio/DATA/CB/pros')

# Glacier name (options: Artesonraju, Ghueshgue, Shallap, Yanamarey)
gla_name = 'Yanamarey'
var_sum  = 'MB'

sel_vars = ['MASK', 'HGT', 'SLOPE', 'ASPECT']

data_path = dir_data / gla_name
nc_files = sorted(data_path.glob('*.nc'))

# -----------------------------------------------------------------------------
# 1) Read and concatenate MB over time
# -----------------------------------------------------------------------------
ds_list = []
for f in nc_files:
    print(f"Reading: {f.name}")
    ds = xr.open_dataset(f)[var_sum]
    ds_list.append(ds)

ds_mb = xr.concat(ds_list, dim="time")

# -----------------------------------------------------------------------------
# 2) Extract static topographic data from first file
# -----------------------------------------------------------------------------
ds_geo = xr.open_dataset(nc_files[0])[sel_vars]

MASK = ds_geo['MASK']

# -----------------------------------------------------------------------------
# 3) Monthly aggregation and masking
# -----------------------------------------------------------------------------
ds_mb_monthly = (
    ds_mb
    .resample(time="1MS")
    .sum()
    .where(MASK == 1)
)

ds_mb_monthly = ds_mb_monthly.to_dataset(name=var_sum)

# -----------------------------------------------------------------------------
# 4) Merge static and dynamic datasets
# -----------------------------------------------------------------------------
ds_merged = xr.merge([ds_geo, ds_mb_monthly])

# -----------------------------------------------------------------------------
# 5) Save merged NetCDF
# -----------------------------------------------------------------------------
out_file = dir_data_out / f"{gla_name}_geodata_mb_monthly.nc"
ds_merged.to_netcdf(out_file)

print(f"\nSaved merged dataset: {out_file}")
print(ds_merged)
