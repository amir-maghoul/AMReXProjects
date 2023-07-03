#ifndef _MYFUNC_H_
#define _MYFUNC_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

void main_main ();

// amrex::Real barenblatt(amrex::Real rsquared, double t, int m, int d);

void advance (amrex::MultiFab& uOld,
              amrex::MultiFab& uNew,
              amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& flux,
              amrex::Real dt,
              amrex::Geometry const& geom, 
              int m);

void initU(amrex::MultiFab& uNew, amrex::Geometry const& geom, int m, double t0);

#endif