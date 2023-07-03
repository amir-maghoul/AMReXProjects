#ifndef _MYKERNEL_H
#define _MYKERNEL_H

#include <AMReX_FArrayBox.H>
#include "barenblatt.h"

/**
 * The initialization kernel for a single cell. In a ParallelFor,
 * this initializes the whole solution array
*/
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void initUKernel(int i, int j, int k, int m, double t0,
            amrex::Array4<amrex::Real> const& uArray,
            amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& ds,
            amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& probLo){


        // Get the real coordinates of the cell-centered point
        amrex::Real x = probLo[0] + (i + 0.5) * ds[0];
        amrex::Real y = probLo[1] + (j + 0.5) * ds[1];

#if (AMREX_SPACEDIM == 2)
        amrex::Real rsquared = ((x-0.5)*(x-0.5)+(y-0.5)*(y-0.5))/0.01;
#elif (AMREX_SPACEDIM == 3)
        amrex::Real z = probLo[2] + (k + 0.5) * ds[2];
        amrex::Real rsquared = ((x-0.5)*(x-0.5)+(y-0.5)*(y-0.5)+(z-0.5)*(z-0.5))/0.01;
#endif
        amrex::Real baren = 0.;
        baren = barenblatt(rsquared, t0, m, AMREX_SPACEDIM);
        uArray(i,j,k) = baren;
        }

#endif

/**
 * Computes the flux in the x direction
*/
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_flux_x (int i, int j, int k,
                     amrex::Array4<amrex::Real> const& fluxx,
                     amrex::Array4<amrex::Real const> const& phi, amrex::Real dxinv, int m)
{
    fluxx(i,j,k) = (std::pow(phi(i,j,k), m)- std::pow(phi(i-1,j,k), m)) * dxinv;
}

/**
 * Computes the flux in y direction
*/
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_flux_y (int i, int j, int k,
                     amrex::Array4<amrex::Real> const& fluxy,
                     amrex::Array4<amrex::Real const> const& phi, amrex::Real dyinv, int m)
{
    fluxy(i,j,k) = (std::pow(phi(i,j,k), m)- std::pow(phi(i,j-1,k), m)) * dyinv;
}

#if (AMREX_SPACEDIM > 2)
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_flux_z (int i, int j, int k,
                     amrex::Array4<amrex::Real> const& fluxz,
                     amrex::Array4<amrex::Real const> const& phi, amrex::Real dzinv, int m)
{
    fluxz(i,j,k) = (std::pow(phi(i,j,k), m) - std::pow(phi(i,j,k-1), m)) * dzinv;
}
#endif

/**
 * Updates the old solution vector with the new values
*/
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void updateU (int i, int j, int k,
                amrex::Array4<amrex::Real const> const& uOldArray,
                amrex::Array4<amrex::Real      > const& uNewArray,
                AMREX_D_DECL(amrex::Array4<amrex::Real const> const& fluxx,
                            amrex::Array4<amrex::Real const> const& fluxy,
                            amrex::Array4<amrex::Real const> const& fluxz),
                amrex::Real dt,
                AMREX_D_DECL(amrex::Real dxinv,
                            amrex::Real dyinv,
                            amrex::Real dzinv))
{
    uNewArray(i,j,k) = uOldArray(i,j,k)
        + dt * dxinv * (fluxx(i+1, j  , k) - fluxx(i, j, k))
        + dt * dyinv * (fluxy(i  , j+1, k) - fluxy(i, j, k))
#if (AMREX_SPACEDIM > 2)
        + dt * dzinv * (fluxz(i  ,j  ,k+1) - fluxz(i,j,k));
#else
        ;
#endif
}