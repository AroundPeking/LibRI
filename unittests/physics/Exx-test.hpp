// ===================
//  Author: Peize Lin
//  date: 2022.06.02
// ===================

#pragma once

#include "RI/physics/Exx.h"
#include <cassert>
#include <complex>

namespace Exx_Test
{
	template<typename Tdata>
	void main(int argc, char *argv[])
	{
		int mpi_init_provide;
		MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_init_provide);

		RI::Exx<int,int,1,Tdata> exx;
		exx.set_parallel(MPI_COMM_WORLD, {{1,{0}},{2,{4}}}, {}, {1});
		exx.set_symmetry(false, {});
	
		exx.set_Cs({}, 1E-4);
		exx.set_Vs({}, 1E-4);
		exx.set_Vs({}, 1E-4, "short");
		exx.set_Ds({}, 1E-4);
		typename RI::Exx<int,int,1,Tdata>::Weighted_Short_Config weighted_short_config;
		weighted_short_config.weighted_short_threshold = 1E-3;
		weighted_short_config.weighted_short_stats_only = true;
		exx.set_weighted_short_config(weighted_short_config);
		assert(exx.weighted_short_config.weighted_short_threshold == weighted_short_config.weighted_short_threshold);
		assert(exx.weighted_short_config.weighted_short_stats_only == weighted_short_config.weighted_short_stats_only);
		assert(exx.weighted_short_config.weighted_short_only == weighted_short_config.weighted_short_only);
		exx.cal_Hs({"","short",""});
		const auto weighted_short_state = RI::lri_get_weighted_short_screen_config(static_cast<const void*>(&exx.lri));
		assert(weighted_short_state.threshold < 0);
		assert(!weighted_short_state.enabled);

		exx.set_dCs({}, 1E-4);
		exx.set_dVs({}, 1E-4);
		exx.cal_force();

		exx.set_dCRs({}, 1E-4);
		exx.set_dVRs({}, 1E-4);
		exx.cal_stress();

		exx.set_Ds_delta({}, 1E-4);
		exx.cal_Hs();

		exx.free_Cs();
		exx.free_Vs();
		exx.free_Ds();
		exx.free_Ds_delta();
		exx.free_dCs();
		exx.free_dVs();
		exx.free_dCRs();
		exx.free_dVRs();

		MPI_Finalize();
	}
}
