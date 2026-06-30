import parcels
import numpy as np

LoggerheadParticle = parcels.Particle.add_variable(
        [
            parcels.Variable('temperature', dtype=np.float32, to_write=True, initial=np.nan),
            parcels.Variable('stranded', dtype=np.bool, to_write=True, initial=False),
        ]
    )


def StokesDriftRK2(particles, fieldset):
    """Stokes drift kernel for particles that are not stranded.


    Kernel Requirements
    ----------
    fieldset :
        - UVStokes: Zonal and meridional Stokes drift velocity at surface [m s-1]

    """
    u, v = fieldset.UVStokes[particles]
    lon1, lat1 = (particles.lon + u * 0.5 * particles.dt, particles.lat + v * 0.5 * particles.dt)
    u, v = fieldset.UV[particles.time + 0.5 * particles.dt, particles.z, lat1, lon1, particles]

    particles.dlon += u * particles.dt
    particles.dlat += v * particles.dt



def WindageRK2(particles, fieldset):
    """Leeway windage kernel using an RK2 scheme for particles that are not stranded.

    Description
    ----------
    A simple windage kernel that applies a linear relative windage

    Kernel Requirements
    ----------
    fieldset :
        - wind_coeff: Windage coefficient [fraction]
        - UV: Ocean velocities [m s-1]
        - UVWind: Wind velocity field at 10m height above sea surface [m s-1]

    """

    ocean_U, ocean_V = fieldset.UV[particles]
    wind_U, wind_V = fieldset.UVWind[particles]
    u = ocean_U + fieldset.wind_coeff * (wind_U - ocean_U)
    v = ocean_V + fieldset.wind_coeff * (wind_V - ocean_V)

    lon1, lat1 = (particles.lon + u * 0.5 * particles.dt, particles.lat + v * 0.5 * particles.dt)

    ocean_U, ocean_V = fieldset.UV[particles.time + 0.5 * particles.dt, particles.z, lat1, lon1, particles]
    wind_U, wind_V = fieldset.UVWind[particles.time + 0.5 * particles.dt, particles.z, lat1, lon1, particles]
    u = ocean_U + fieldset.wind_coeff * (wind_U - ocean_U)
    v = ocean_V + fieldset.wind_coeff * (wind_V - ocean_V)

    particles.dlon += u * particles.dt
    particles.dlat += v * particles.dt


def Stranding(particles, fieldset):
    """Data-based stranding kernel.

    Description
    ----------
    Kernel that determines which particles are stranded under the condition
    that U or V == 0. Transport of stranded particles is set to zero.

    Kernel Requirements
    ----------
    particle :
        - stranded: Boolean whether particle is straned (1) or not (0)
    fieldset :
        - UV: Ocean velocities [m s-1]

    Order of Operations:
    ----------
        At the end of physical kernels. Otherwise dlon and dlat will be updated again.

    """

    u, v = fieldset.UV[particles]

    # particles.stranded = np.where(, True, particles.stranded)
    stranded = (u == 0.0) | (v == 0.0)
    if np.any(stranded):
        ptcls_stranded = particles[stranded]
        ptcls_stranded.stranded = True
        fieldset.output_file.write(ptcls_stranded, ptcls_stranded.time[0], fieldset=fieldset)
        ptcls_stranded.state = parcels.StatusCode.Delete


def UnbeachingBySampling(particles, fieldset):
    """Unbeaching particles by sampling the velocity field and moving in the direction of velocities!
    Taken from https://github.com/Parcels-code/NECCTONsimulations/blob/main/kernels.py#L973
    """
    # Measure the velocity field at the final particle location
    # Note, we've already handled the surface and bathymetry boundary conditions, so the particle is within the water column
    (vel_u, vel_v) = fieldset.UV[
        particles.time,
        particles.z + particles.dz,
        particles.lat + particles.dlat,
        particles.lon + particles.dlon
    ]

    # Particles whose future velocity is (nearly) zero will be on land
    stuck_ptcls = particles[np.sqrt(vel_u**2 + vel_v**2) < 1e-9]
    if np.any(stuck_ptcls.lon):
        unbeach_U = 1. / (1852. * 60. * np.cos(stuck_ptcls.lat * np.pi / 180.)) # Convert 1m/2s to degrees/s at the particle latitude in zonal direction
        unbeach_V = 1. / (1852. * 60.) # Convert 1m/2s to degrees/s in meridional direction

        displacement = 1./8. # Degree displacement to sample the velocity field
        test_vel = np.zeros((len(stuck_ptcls.lon)))
        DX, DY = np.zeros((len(stuck_ptcls.lon))), np.zeros((len(stuck_ptcls.lon)))
        for dx, dy in zip([-1, 1, 0, 0, -1, 1, -1, 1], [0, 0, 1, -1, 1, 1, -1, -1]):
            (U_test, V_test) = fieldset.UV[
                stuck_ptcls.time,
                stuck_ptcls.z + stuck_ptcls.dz,
                stuck_ptcls.lat + stuck_ptcls.dlat + dy * displacement,
                stuck_ptcls.lon + stuck_ptcls.dlon + dx * displacement
            ]
            idx = U_test**2 + V_test**2 > test_vel
            if np.any(idx):
                test_vel[idx] = U_test[idx]**2 + V_test[idx]**2
                DX[idx], DY[idx] = dx, dy

        if np.any(test_vel > 1e-9):
            dlon = DX * unbeach_U * np.abs(stuck_ptcls.dt)
            dlat = DY * unbeach_V * np.abs(stuck_ptcls.dt)
            stuck_ptcls.dlon += dlon
            stuck_ptcls.dlat += dlat
