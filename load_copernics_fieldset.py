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

    DATASET_IDs_BY_GRID: list[tuple[str, Grid]] = [
        (
            "physics",
            [
                ("cmems_mod_glo_phy_my_0.083deg_P1D-m", ("uo", "vo", "thetao",)),
            ],
        ),
        (
            "waves",
            [
                ("cmems_mod_glo_wav_my_0.2deg_PT3H-i", ("VSDX", "VSDY")),
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
            minimum_longitude=-20,
            maximum_longitude=10,
            minimum_latitude=40,
            maximum_latitude=65,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            minimum_depth=0.5,
            maximum_depth=0.5,
        )
    )

    datasets = {}
    fset = []
    for name, grid_datasets in DATASET_IDs_BY_GRID:
        datasets[name] = {}
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
        datasets[name].to_netcdf(f"copernicus_{name}.nc", mode="w")

    for name in datasets.keys():
        ds = xr.open_dataset(f"copernicus_{name}.nc")
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

    fieldset = fset[0]
    for f in fset[1:]:
        fieldset += f

    fieldset.to_windowed_arrays()

    return fieldset
