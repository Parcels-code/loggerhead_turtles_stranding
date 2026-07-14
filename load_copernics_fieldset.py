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
            ],
        ),
    ]

    copernicus_kwargs = (
        dict(
            minimum_longitude=-5,
            maximum_longitude=10,
            minimum_latitude=45,
            maximum_latitude=58,
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
        if name == "physics":
            ds_ab = xr.open_dataset("/home/evansebill/loggerhead_turtles_stranding/anti_beaching_NWES_Met.nc")
            ds_ab = ds_ab.rename({"lon": "longitude", "lat": "latitude", "dispU": "Uab", "dispV": "Vab"})
            datasets_list.append(ds_ab)
        ds = xr.merge(datasets_list)

        # TODO Should this processing go to copernicusmarine_to_sgrid?
        ds = ds.rename({"uo": "U", "vo": "V"})
        ds["U"] = ds["U"].fillna(0)
        ds["V"] = ds["V"].fillna(0)

        datasets[name] = parcels.convert.copernicusmarine_to_sgrid(
            fields={name: da for name, da in ds.data_vars.items()}
        )
        fset.append(
            parcels.FieldSet.from_sgrid_conventions(
                datasets[name],
                vector_fields={"UV": ("U", "V"), "UVab": ("Uab", "Vab")}
            )
        )

    fieldset = fset[0]
    for f in fset[1:]:
        fieldset += f

    fieldset.to_windowed_arrays()

    return fieldset
