import os
import psutil

import copernicusmarine
import parcels
import xarray as xr
import numpy as np

import zarr
from zarr.experimental.cache_store import CacheStore

ModelId = str
UsedFields = tuple[str, ...]
Grid = list[tuple[ModelId, UsedFields]]

def create_fieldset(startdate, enddate):

    startdate = np.datetime64(startdate)
    enddate = np.datetime64(enddate)
    start_datetime = np.datetime_as_string(startdate, unit="s")
    end_datetime = np.datetime_as_string(enddate, unit="s")
    start_ymd = np.datetime_as_string(startdate, unit="D").replace("-", "")
    end_ymd = np.datetime_as_string(enddate, unit="D").replace("-", "")

    DATASET_IDs = [
        "cmems_mod_nws_phy-cur_anfc_1.5km-2D_PT1H-i", # TODO use cmems_mod_nws_phy-cur_anfc_1.5km-2D_PT15M-i
    #     # "cmems_mod_nws_phy-sst_anfc_1.5km-2D_PT1H-i",
    #     # "cmems_mod_nws_wav_anfc_1.5km_PT1H-i",
    #     # "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H",
    ]

    DATASET_IDs_BY_GRID: list[tuple[str, Grid]] = [
        (
            "physics",
            [
                ("cmems_mod_nws_phy-cur_anfc_1.5km-2D_PT1H-i", ("uo", "vo")), # TODO use cmems_mod_nws_phy-cur_anfc_1.5km-2D_PT15M-i
                ("cmems_mod_nws_phy-sst_anfc_1.5km-2D_PT1H-i", ("thetao",)),
            ],
        ),
        (
            "waves",
            [
                ("MetO-NWS-WAV-RAN", ("VSDX", "VSDY")),
            ],
        ),
        (
            "wind",
            [
                ("cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H", ("northward_wind", "eastward_wind")),
            ],
        ),
    ]

    copernicus_kwargs = (
        dict(
            minimum_longitude=-1,
            maximum_longitude=7,
            minimum_latitude=50,
            maximum_latitude=55,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            minimum_depth=0.5,
            maximum_depth=0.5,
        )
    )

    datasets = {}
    fset = []
    for name, grid_datasets in DATASET_IDs_BY_GRID:
        copernicusmarine.login()
        datasets_list = [
            copernicusmarine.open_dataset(id_, **copernicus_kwargs)[list(used_vars)]
            for id_, used_vars in grid_datasets
        ]
        # TODO Should this processing go to copernicusmarine_to_sgrid?
        for i in range(len(datasets_list)):
            if "depth" not in datasets_list[i].dims:
                datasets_list[i] = datasets_list[i].expand_dims(
                    dim={"depth": [0]}, axis=1
                )
        ds = xr.merge(datasets_list)

        if "uo" in ds.data_vars and "vo" in ds.data_vars:
            ds = ds.rename({"uo": "U", "vo": "V"})
            vector_fields = {"UV": ("U", "V")}
        elif "northward_wind" in ds.data_vars and "eastward_wind" in ds.data_vars:
            vector_fields = {"UVWind": ("northward_wind", "eastward_wind")}
        elif "VSDX" in ds.data_vars and "VSDY" in ds.data_vars:
            vector_fields = {"UVStokes": ("VSDX", "VSDY")}
        else:
            vector_fields = {}

        for var in ds.data_vars:
            ds[var] = ds[var].fillna(0)

        datasets[name] = parcels.convert.copernicusmarine_to_sgrid(
            fields={name: da for name, da in ds.data_vars.items()}
        )

### NOTE CODE BELOW IS BECAUSE STREAMING FROM COPERNICUSMARINE IS VERY SLOW (2 HRS instead of 2 minutes for a simulation).
### CAN BE REMOVED WHEN STREAMING IS FAST AGAIN
        datasets[name].to_netcdf(f"copernicus_{name}_{start_ymd}_{end_ymd}.nc")

    for name in datasets.keys():
        ds = xr.open_dataset(f"copernicus_{name}_{start_ymd}_{end_ymd}.nc")
        if name == "physics":
            vector_fields = {"UV": ("U", "V")}
        elif name == "waves":
            vector_fields = {"UVStokes": ("VSDX", "VSDY")}
        elif name == "wind":
            vector_fields = {"UVWind": ("northward_wind", "eastward_wind")}
        else:
            vector_fields = {}
        fset.append(
            parcels.FieldSet.from_sgrid_conventions(
                ds,
                vector_fields=vector_fields
            )
        )
### REPLACE WITH CODE BELOW

        # fset.append(
        #     parcels.FieldSet.from_sgrid_conventions(
        #         datasets[name],
        #         vector_fields=vector_fields
        #     )
        # )
### END NOTE

    ds_antibeaching = xr.open_dataset("/home/evansebill/loggerhead_turtles_stranding/anti_beaching_NWES_Met.nc")
    ds_antibeaching = ds_antibeaching.rename({"dispU": "U_antibeaching", "dispV": "V_antibeaching"})
    ds_antibeaching["lon"].attrs["units"] = "degrees_east"
    ds_antibeaching["lat"].attrs["units"] = "degrees_north"
    ds_antibeaching.load()

    ds = parcels.convert.copernicusmarine_to_sgrid(
        fields={
            "U_antibeaching": ds_antibeaching["U_antibeaching"],
            "V_antibeaching": ds_antibeaching["V_antibeaching"]}
    )
    fset_ab = parcels.FieldSet.from_sgrid_conventions(ds, vector_fields={"UV_antibeaching": ("U_antibeaching", "V_antibeaching")})
    fset.append(fset_ab)

    fieldset = fset[0]
    for f in fset[1:]:
        fieldset += f

    fieldset.to_windowed_arrays()
    fieldset.describe()

    return fieldset
