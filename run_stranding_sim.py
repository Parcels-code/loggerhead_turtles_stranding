import datetime

import numpy as np
import pandas as pd
import polars as pl

import parcels

import load_copernics_fieldset

df_in = pd.read_excel("stranding_locations.xlsx")

def AdvectionRK2_withunbeaching(particles, fieldset):  # pragma: no cover
    """Advection of particles using second-order Runge-Kutta integration."""
    particles.temperature = fieldset.thetao[particles]

    (u1, v1) = fieldset.UV[particles]
    (uw1, vw1) = fieldset.UVWind[particles]
    (us1, vs1) = fieldset.UVStokes[particles]
    x1 = particles.x + (u1 + uw1 * fieldset.wind_coeff + us1) * 0.5 * particles.dt
    y1 = particles.y + (v1 + vw1 * fieldset.wind_coeff + vs1) * 0.5 * particles.dt

    (u2, v2) = fieldset.UV[particles.t + 0.5 * particles.dt, particles.z, y1, x1, particles]
    (uw2, vw2) = fieldset.UVWind[particles.t + 0.5 * particles.dt, particles.z, y1, x1, particles]
    (us2, vs2) = fieldset.UVStokes[particles.t + 0.5 * particles.dt, particles.z, y1, x1, particles]

    # Particles whose temperature is (nearly) zero are on land
    stuck_ptcls = particles[particles.temperature < 1e-1]

    if np.any(stuck_ptcls.x):
        unbeach_U = 1. / (1852. * 60. * np.cos(stuck_ptcls.y * np.pi / 180.)) # Convert 1m/s to degrees/s at the particle latitude in zonal direction
        unbeach_V = 1. / (1852. * 60.) # Convert 1m/2s to degrees/s in meridional direction

        displacement = 1./12. # Degree displacement to sample the velocity field
        test_temp = np.zeros((len(stuck_ptcls.x)))
        DX, DY = np.zeros((len(stuck_ptcls.x))), np.zeros((len(stuck_ptcls.x)))
        for dx, dy in zip([-1, 1, 0, 0, -1, 1, -1, 1], [0, 0, 1, -1, 1, 1, -1, -1]):
            temp_test = fieldset.thetao[
                stuck_ptcls.t,
                stuck_ptcls.z,
                stuck_ptcls.y + dy * displacement,
                stuck_ptcls.x + dx * displacement
            ]
            idx = temp_test > test_temp
            if np.any(idx):
                test_temp[idx] = temp_test[idx]
                DX[idx], DY[idx] = dx, dy

        if np.any(test_temp > 1e-1):
            dlon = DX * unbeach_U * np.abs(stuck_ptcls.dt)
            dlat = DY * unbeach_V * np.abs(stuck_ptcls.dt)
            stuck_ptcls.dx += dlon
            stuck_ptcls.dy += dlat

    particles.dx += (u2 + us2 + uw2 * fieldset.wind_coeff) * particles.dt
    particles.dy += (v2 + vs2 + vw2 * fieldset.wind_coeff) * particles.dt


def run_sim(beach_date, beach_lat, beach_lon, beach_id):
    runtime = datetime.timedelta(days=120)

    fieldset = load_copernics_fieldset.create_fieldset(startdate=beach_date-runtime, enddate=beach_date)

    # Set temperature interpolation
    fieldset.thetao.interp_method = parcels.interpolators.XLinearInvdistLandTracer()

    # Windage coefficient [fraction]
    fieldset.wind_coeff = 0.01

    N = 100
    lons = beach_lon + np.random.uniform(-0.08, 0.08, size=N)
    lats = beach_lat + np.random.uniform(-0.08, 0.08, size=N)

    pfile = parcels.ParticleFile(
        f"Loggerhead_Simulation_{beach_id}.parquet",
        outputdt=np.timedelta64(1, 'h'),
        mode="w",
    )

    tempvar = parcels.Variable('temperature', initial=np.nan),
    LoggerheadParticle = parcels.Particle.add_variable([tempvar])

    pset = parcels.ParticleSet(
        fieldset=fieldset,
        pclass=LoggerheadParticle,
        x=lons,
        y=lats,
        t=np.datetime64(beach_date),
    )

    kernels = [AdvectionRK2_withunbeaching]

    pset.execute(
        kernels=kernels,
        runtime=runtime,
        dt=-np.timedelta64(10, 'm'),
        output_file=pfile,
    )

for index, row in df_in.iterrows():
    location = row["Locatie"]
    beach_id = row["ID"]
    beach_date = row["Datum"]
    if pd.isna(location):
        print(f"Invalid location for location {beach_id}: {location}")
        continue
    beach_lat = float(location.split(",")[0].strip())
    beach_lon = float(location.split(",")[1].strip())
    print(f"Stranding location {beach_id}: Latitude: {beach_lat}, Longitude: {beach_lon} on {beach_date}")

    run_sim(beach_date, beach_lat, beach_lon, beach_id)
