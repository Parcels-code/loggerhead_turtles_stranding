import datetime
import glob
import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from scipy import interpolate

import cartopy.crs as ccrs
import cartopy.feature

import parcels

import load_copernics_fieldset
from kernels import WindageRK2, Stranding, LoggerheadParticle, StokesDriftRK2, UnbeachingBySampling

df_in = pd.read_excel("stranding_locations.xlsx")
df_in

offsets = {  # TODO check if still needed with UnbeachingBySampling kernel
    "CC26": (0.02, 0),
    "CC32": (0, -0.02),
    "CC23": (0.015, 0),
    "CC33": (0.01, 0),
    "CC31": (0, -0.02),
}

def run_sim(beach_date, beach_lat, beach_lon, beach_id):
    fieldset = load_copernics_fieldset.create_fieldset(startdate=beach_date-datetime.timedelta(days=30), enddate=beach_date)

    # Windage coefficient [fraction]
    fieldset.wind_coeff = 0.01

    N = 10
    lons = beach_lon + np.random.uniform(-0.01, 0.01, size=N)
    lats = beach_lat + np.random.uniform(-0.01, 0.01, size=N)

    pfile = parcels.ParticleFile(
        f"Loggerhead_Simulation_{beach_id}.parquet",
        outputdt=np.timedelta64(1, 'h'),
        mode="w",
    )
    # fieldset.output_file = pfile  # for writing stranded particles

    pset = parcels.ParticleSet(
        fieldset=fieldset,
        pclass=LoggerheadParticle,
        x=lons,
        y=lats,
        t=np.datetime64(beach_date),
    )

    def AdvectionRK2_withunbeaching(particles, fieldset):  # pragma: no cover
        """Advection of particles using second-order Runge-Kutta integration."""
        (u1, v1) = fieldset.UV[particles]
        x1, y1 = (particles.x + u1 * 0.5 * particles.dt, particles.y + v1 * 0.5 * particles.dt)
        (u2, v2) = fieldset.UV[particles.t + 0.5 * particles.dt, particles.z, y1, x1, particles]

        uab, vab = fieldset.UVab[particles.t + 0.5 * particles.dt, particles.z, y1, x1, particles]
        particles.dx += (u2 - uab) * particles.dt
        particles.dy += (v2 - vab) * particles.dt

    kernels = [
        # Stranding,
        # parcels.kernels.AdvectionRK2,
        # WindageRK2,
        # StokesDriftRK2,
        # UnbeachingBySampling,
        AdvectionRK2_withunbeaching
    ]

    pset.execute(
        kernels=kernels,
        runtime=np.timedelta64(10, 'D'),
        dt=-np.timedelta64(10, 'm'),
        output_file=pfile,
    )
    gc.collect()

for index, row in df_in.iterrows():
    location = row["Locatie"]
    beach_id = row["ID"]
    beach_date = row["Datum"]
    if beach_date < datetime.datetime(2024, 4, 1):
        print(f"Invalid date for location {beach_id}: {beach_date}")
        continue
    if pd.isna(location):
        print(f"Invalid location for location {beach_id}: {location}")
        continue
    offset = (0, 0) if beach_id not in offsets.keys() else offsets[beach_id]
    beach_lat = float(location.split(",")[0].strip()) + offset[0]
    beach_lon = float(location.split(",")[1].strip()) + offset[1]
    print(f"Stranding location {beach_id}: Latitude: {beach_lat}, Longitude: {beach_lon} on {beach_date}")

    run_sim(beach_date, beach_lat, beach_lon, beach_id)