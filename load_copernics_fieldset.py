import copernicusmarine
import os
import parcels
import xarray as xr
import numpy as np

def create_fieldset(startdate, enddate):
    # TODO make this on-the-fly access instead of downloading the file first and then loading it

    startdate = np.datetime64(startdate)
    enddate = np.datetime64(enddate)
    start_datetime = np.datetime_as_string(startdate, unit="s")
    end_datetime = np.datetime_as_string(enddate, unit="s")
    start_ymd = np.datetime_as_string(startdate, unit="D").replace("-", "")
    end_ymd = np.datetime_as_string(enddate, unit="D").replace("-", "")
    dirname = "copernicus_marine_data"
    os.makedirs(dirname, exist_ok=True)

    DATASET_IDs = [
        "cmems_mod_nws_phy-cur_anfc_1.5km-2D_PT1H-i", # TODO use cmems_mod_nws_phy-cur_anfc_1.5km-2D_PT15M-i
        "cmems_mod_nws_phy-sst_anfc_1.5km-2D_PT1H-i",
        # "cmems_mod_nws_wav_anfc_1.5km_PT1H-i",
        # "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H",
    ]
    variables = [["uo", "vo"], ["thetao"], ["VSDX", "VSDY", "VTPK"], ["northward_wind", "eastward_wind"]]
    filenames = ["curr.zarr", "thetao.zarr", "stokes.zarr", "wind.zarr"]
    for i in range(len(DATASET_IDs)):

        filename = os.path.join(dirname, f"copernicusmarine_{start_ymd}_{end_ymd}_{filenames[i]}")
        if not os.path.exists(filename):
            copernicusmarine.login()

            copernicusmarine.subset(
                dataset_id=DATASET_IDs[i],
                variables=variables[i],
                minimum_longitude=-5,
                maximum_longitude=10,
                minimum_latitude=45,
                maximum_latitude=58,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                minimum_depth=0.5,
                maximum_depth=0.5,
                output_filename=filename,
            )

    fields = []

    for i in range(len(filenames)):
        filename = os.path.join(dirname, f"copernicusmarine_{start_ymd}_{end_ymd}_{filenames[i]}")

        # print(f"Loading {filename}")
        ds_fields = xr.open_mfdataset(filename, combine="by_coords")
        try:
            # TODO check how we can get good performance without loading full dataset in memory
            ds_fields.load()  # load the dataset into memory

            ds_fields = ds_fields.expand_dims(dim={"depth": [0]})
            # TODO clean up this part, maybe make it more generic and less hardcoded
            if "cur" in filename:
                flds = {"U": ds_fields["uo"], "V": ds_fields["vo"]}
            elif "thetao" in filename:
                flds = {"thetao": ds_fields["thetao"]}
            elif "stokes" in filename:
                flds = {"VSDX": ds_fields["VSDX"], "VSDY": ds_fields["VSDY"], "VTPK": ds_fields["VTPK"]}
            elif "wind" in filename:
                flds = {"northward_wind": ds_fields["northward_wind"], "eastward_wind": ds_fields["eastward_wind"]}

            ds_fset = parcels.convert.copernicusmarine_to_sgrid(fields=flds)
            fset = parcels.FieldSet.from_sgrid_conventions(ds_fset, mesh="spherical")

            for fld in fset.fields.values():
                fields.append(fld)
        finally:
            ds_fields.close()
    fieldset = parcels.FieldSet(fields)
    # fieldset.UV.interp_method = parcels.interpolators.XFreeslip

    # # making vector fields of wind and stokes
    # fieldset.add_field(parcels.VectorField("UVStokes", fieldset.VSDX, fieldset.VSDY, vector_interp_method=parcels.interpolators.XFreeslip))
    # fieldset.add_field(parcels.VectorField("UVWind", fieldset.northward_wind, fieldset.eastward_wind, vector_interp_method=parcels.interpolators.XFreeslip))

    # setting NaN values to 0 for vector fields
    for field in ["U", "V"]: #, "VSDX", "VSDY", "northward_wind", "eastward_wind"]:
        getattr(fieldset, field).data = getattr(fieldset, field).data.fillna(0)

    fieldset.thetao.data = fieldset.thetao.data.fillna(0)
    # fieldset.VTPK.data = fieldset.VTPK.data.fillna(0)

    #setting other interpolators to XLinearInvdistLandTracer,
    for field in [fieldset.thetao]: #, fieldset.VTPK]:
        field.interp_method = parcels.interpolators.XLinearInvdistLandTracer

    return fieldset
