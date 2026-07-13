from pathlib import Path
import xarray as xr

# -----------------------------------------------------------------------------
# Paths and configuration
# -----------------------------------------------------------------------------
dir_data_cpy   = Path('/home/lacrio/DATA/CB/pros')
dir_data_era5  = Path('/home/lacrio/DATA/ERA5Land/raw')

# Glacier name (options: Artesonraju, Ghueshgue, Shallap, Yanamarey)
gla_name = 'Yanamarey'


ds_cpy  = xr.open_dataset( dir_data_cpy / f"{gla_name}_monthly.nc")
print(ds_cpy)
ds_era5l     = xr.open_dataset(dir_data_era5 / "era5_monthly_averaged_data_TRO.nc")
#ds_era5l_hgt = xr.open_dataset(dir_data_era5 / "era5_geopotential_pressure.nc")



ds_era5_on_cpy = ds_era5l.interp(
    latitude=ds_cpy.lat,
    longitude=ds_cpy.lon,
    method="nearest"   # or "nearest" if you prefer
)

print(ds_era5_on_cpy) 

out_file = dir_data_cpy / f"{gla_name}_cpy_era5_l.nc"

ds_merge = xr.merge([ds_cpy, ds_era5_on_cpy])

ds_merge.to_netcdf(out_file)

print(ds_merge)
