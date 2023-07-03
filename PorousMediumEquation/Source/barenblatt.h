#ifndef _BARENBLATT_H_
#define _BARENBLATT_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

inline amrex::Real barenblatt(amrex::Real rsquared, double t, int m, int d){

    double d_2 = 2.0/d;

    double alpha = 1.0/(m-1.0+d_2);
    double t_a = std::pow(t, alpha);

    double bigTerm = (1.0 - (alpha*(m-1)*rsquared/(2*d*m*(std::pow(t_a, d_2)))));

    if (bigTerm <= 0){
        return 0.0;
    }
    else {
        bigTerm = std::pow(bigTerm, (1.0/(m-1)));
        bigTerm = bigTerm/t_a;
        return bigTerm;
    }
};


#endif