// ====================
//  Author: Peize Lin
//  Date:   2022.07.25
// ====================

#pragma once

#include "RI/physics/RPA.h"
#include <complex>

namespace RPA_Test
{
	template<typename Tdata>
	void main(int argc, char *argv[])
	{
		int mpi_init_provide;
		MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_init_provide);

		RPA<int,int,1,std::complex<double>> rpa;
		rpa.set_parallel(MPI_COMM_WORLD, {{1,{0}},{2,{4}}}, {}, {1});
		rpa.set_Cs({}, 0);
		rpa.cal_chi0s({}, {}, 0);

		MPI_Finalize();
	}
}