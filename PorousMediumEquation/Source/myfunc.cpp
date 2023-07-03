#include "myfunc.H"
#include "mykernel.H"
#include <cmath>

using namespace amrex;

/**
 * Forward propogation of values in time
*/
void advance (MultiFab& uOld,
              MultiFab& uNew,
              Array<MultiFab, AMREX_SPACEDIM>& flux,
              Real dt,
              Geometry const& geom,
              int m)
{
    BL_PROFILE("advance");

    // Fill the ghost cells of each grid from the other grids
    // includes periodic domain boundaries.
    // There are no physical domain boundaries to fill in this example.
    uOld.FillBoundary(geom.periodicity());

    //
    // Note that this simple example is not optimized.
    // The following two MFIter loops could be merged
    // and we do not have to use flux MultiFab.
    //
    // =======================================================

    // This example supports both 2D and 3D.  Otherwise,
    // we would not need to use AMREX_D_TERM.
    AMREX_D_TERM(const Real dxinv = geom.InvCellSize(0);,
                 const Real dyinv = geom.InvCellSize(1);,
                 const Real dzinv = geom.InvCellSize(2););

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    // Compute fluxes one grid at a time
    for ( MFIter mfi(uOld); mfi.isValid(); ++mfi )
    {
        const Box& xbx = mfi.nodaltilebox(0);
        const Box& ybx = mfi.nodaltilebox(1);
        auto const& fluxx = flux[0].array(mfi);
        auto const& fluxy = flux[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        const Box& zbx = mfi.nodaltilebox(2);
        auto const& fluxz = flux[2].array(mfi);
#endif
        auto const& intermediateU = uOld.const_array(mfi);

        amrex::ParallelFor(xbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            compute_flux_x(i, j, k, fluxx, intermediateU, dxinv, m);
        });

        amrex::ParallelFor(ybx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            compute_flux_y(i, j, k, fluxy, intermediateU, dyinv, m);
        });

#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            compute_flux_z(i,j,k,fluxz,intermediateU,dzinv);
        });
#endif
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    // Advance the solution one grid at a time
    for ( MFIter mfi(uOld); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& fluxx = flux[0].const_array(mfi);
        auto const& fluxy = flux[1].const_array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& fluxz = flux[2].const_array(mfi);
#endif
        auto const& uOldArray = uOld.const_array(mfi);
        auto const& uNewArray = uNew.array(mfi);

        amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            updateU(i, j, k, uOldArray, uNewArray,
                       AMREX_D_DECL(fluxx,fluxy,fluxz), dt,
                       AMREX_D_DECL(dxinv,dyinv,dzinv));
        });
    }
}


/**
 * Initializes the solution vectors
*/
void initU(amrex::MultiFab& uNew, amrex::Geometry const& geom, int m, double t0){

    GpuArray<Real,AMREX_SPACEDIM> ds = geom.CellSizeArray();
    GpuArray<Real,AMREX_SPACEDIM> probLo = geom.ProbLoArray();
    // =======================================
    // Initialize uNew
    // MFIter = MultiFab Iterator

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(uNew); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();
        auto const& phiNew = uNew.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            initUKernel(i, j, k, m, t0, phiNew, ds, probLo);
        });
    }
}


