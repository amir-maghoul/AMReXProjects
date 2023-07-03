#include <AMReX_Gpu.H>
#include <AMReX_Utility.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include "myfunc.H"
#include "barenblatt.h"


int main(int argc, char* argv[]){

    amrex::Initialize(argc, argv);

    main_main();

    amrex::Finalize();
    return 0;

}

void main_main(){
    // What time is it now?  We'll use this to compute total run time.
    auto strt_time = amrex::ParallelDescriptor::second();

    /**********************************************************
     * SIMULATION PARAMETERS
    ***********************************************************/

    int Nghost = 1;                             // Nghost = number of ghost cells for each array
    int Ncomp = 1;                              // Ncomp = number of components for each array
    int Ncell;                                  // number of cells on each side of the domain
    int MaxGridSize;                            // size of each box (or grid)
    int Nsteps;                                 // total steps in simulation
    int PlotInt;                                // how often to write a plotfile
    amrex::Real dt;


    // inputs parameters
    {
        // ParmParse is way of reading inputs from the inputs file
        // pp.get means we require the inputs file to have it
        // pp.query means we optionally need the inputs file to have it - but we must supply a default here
        amrex::ParmParse pp;
        // We need to get n_cell from the inputs file - this is the number of cells on each side of
        //   a square (or cubic) domain.
        pp.get("n_cell", Ncell);
        // The domain is broken into boxes of size max_grid_size
        pp.get("max_grid_size",MaxGridSize);
        // Default nsteps to 10, allow us to set it to something else in the inputs file
        Nsteps = 10;
        pp.query("nsteps",Nsteps);
        // Default plot_int to -1, allow us to set it to something else in the inputs file
        //  If plot_int < 0 then no plot files will be written
        PlotInt = -1;
        pp.query("plot_int",PlotInt);
        pp.get("dt", dt);
    }


    /***************************************************************
     * SIMULATION STEP
    ****************************************************************/
    amrex::Geometry geom;
    amrex::BoxArray ba;

    // Domain limits
    amrex::IntVect dom_lo(AMREX_D_DECL(0.       , 0.          , 0.));
    amrex::IntVect dom_hi(AMREX_D_DECL(Ncell - 1, Ncell - 1   , Ncell - 1));

    // Initialize boxarray "ba" from the single box "domain"
    amrex::Box domain(dom_lo, dom_hi);                          
    ba.define(domain);
    ba.maxSize(MaxGridSize);                                          //Define maximum of each boxes in the grid



    // Defining a concrete real box
    // amrex::RealBox real_box({AMREX_D_DECL(-1., -1., -1.)}, {AMREX_D_DECL(1., 1., 1.)});
    amrex::RealBox real_box({AMREX_D_DECL(0., 0., 0.)}, 
                            {AMREX_D_DECL(1.0, 1.0, 1.0)});

    // periodic in all direction
    amrex::Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1,1,1)};

    // Update geometry using the above BoxArray and RealBox
    geom.define(domain, real_box, amrex::CoordSys::cartesian, is_periodic);
    // amrex::Geometry geom(domain, &real_box);
    
    // Extract ds from geometry
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> ds = geom.CellSizeArray();

    // How Boxes are distrubuted among MPI processes
    amrex::DistributionMapping dm(ba);

    // Two Multifabs for new and old values of U in the simulation
    amrex::MultiFab uOld(ba, dm, Ncomp, Nghost);
    amrex::MultiFab uNew(ba, dm, Ncomp, Nghost);

    // *********************************************************
    // INITIALIZE DATA
    // *********************************************************

    int m = 2;
    amrex::Real time = 0.01;
    initU(uNew, geom, m, time);



    /**********************************************************
     * SIMULATION TEMPORAL PARAMETERS
    ***********************************************************/
    // time = starting time in the simulation

    // amrex::Real cfl = 1;
    // amrex::Real coeff = AMREX_D_TERM(   1./(ds[0]*ds[0]),
    //                            + 1./(ds[1]*ds[1]),
    //                            + 1./(ds[2]*ds[2]) );
    // amrex::Real dt = cfl/(2.0*coeff);
    // amrex::Real dt = (endTime - time)/(Nsteps);


    if (PlotInt > 0)
    {
        int step = 0;
        const std::string& pltfile = amrex::Concatenate("plt",step,5);
        WriteSingleLevelPlotfile(pltfile, uNew, {"u"}, geom, time, 0);
    }

    // build the flux multifabs
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> flux;
    // loop over number of dimensions for calculate the flux
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++) //dir for direction
    {
        // flux(direction) has one component, zero ghost cells, and is nodal in direction dir
        amrex::BoxArray edgeBa = ba;
        edgeBa.surroundingNodes(dir);
        flux[dir].define(edgeBa, dm, 1, 0);
    }


    // Time solver
    for (int n = 1; n <= Nsteps; ++n)
    {
        amrex::MultiFab::Copy(uOld, uNew, 0, 0, 1, 0);

        // new_phi = old_phi + dt * (something)
        advance(uOld, uNew, flux, dt, geom, m);
        time = time + dt;

        // Tell the I/O Processor to write out which step we're doing
        amrex::Print() << "Advanced step " << n << "\n";

        //Write a plotfile of the current data (plot_int was defined in the inputs file)
        if (PlotInt > 0 && n%PlotInt == 0)
        {
            const std::string& pltfile = amrex::Concatenate("plt",n,5);
            WriteSingleLevelPlotfile(pltfile, uNew, {"u"}, geom, time, n);
        }
    }

    // Call the timer again and compute the maximum difference between the start time and stop time
    //   over all processors
    auto stop_time = amrex::ParallelDescriptor::second() - strt_time;
    const int IOProc = amrex::ParallelDescriptor::IOProcessorNumber();
    amrex::ParallelDescriptor::ReduceRealMax(stop_time,IOProc);

    // Tell the I/O Processor to write out the "run time"
    amrex::Print() << "Run time = " << stop_time << std::endl;



}