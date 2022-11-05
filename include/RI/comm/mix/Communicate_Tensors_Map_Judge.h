//=======================
// AUTHOR : Peize Lin
// DATE :   2022-07-20
//=======================

#pragma once

#include "Communicate_Tensors_Map.h"
#include "../../global/Cereal_Types.h"
#include <Comm/example/Communicate_Map-2.h>
#include "../example/Communicate_Map_Period.h"

#include <map>
#include <mpi.h>

namespace Communicate_Tensors_Map_Judge
{
	template<typename TA, typename Tdata>
	std::map<TA,Tensor<Tdata>>
	comm_map(
		const MPI_Comm &mpi_comm,
		const std::map<TA,Tensor<Tdata>> &Ds_in,
		const std::set<TA> &s)
	{
		Comm::Communicate_Map::Judge_Map<TA> judge;
		judge.s = s;
		return Communicate_Tensors_Map::comm_map(mpi_comm, Ds_in, judge);
	}

	template<typename TA, typename TAC, typename Tdata>
	std::map<TA,std::map<TAC,Tensor<Tdata>>>
	comm_map2(
		const MPI_Comm &mpi_comm,
		const std::map<TA,std::map<TAC,Tensor<Tdata>>> &Ds_in,
		const std::set<TA> &s0, const std::set<TAC> &s1)
	{
		Comm::Communicate_Map::Judge_Map2<TA,TAC> judge;
		judge.s0 = s0;
		judge.s1 = s1;
		return Communicate_Tensors_Map::comm_map2(mpi_comm, Ds_in, judge);
	}

	template<typename TA, typename TAC, typename Tdata>
	std::map<TA,std::map<TAC,std::map<TAC,Tensor<Tdata>>>>
	comm_map3(
		const MPI_Comm &mpi_comm,
		const std::map<TA,std::map<TAC,std::map<TAC,Tensor<Tdata>>>> &Ds_in,
		const std::set<TA> &s0, const std::set<TAC> &s1, const std::set<TAC> &s2)
	{
		Comm::Communicate_Map::Judge_Map3<TA,TAC,TAC> judge;
		judge.s0 = s0;
		judge.s1 = s1;
		judge.s2 = s2;
		return Communicate_Tensors_Map::comm_map3(mpi_comm, Ds_in, judge);
	}

	template<typename TA, typename TAC, typename Tdata>
	std::map<TA,std::map<TAC,Tensor<Tdata>>>
	comm_map2_first(
		const MPI_Comm &mpi_comm,
		const std::map<TA,std::map<TAC,Tensor<Tdata>>> &Ds_in,
		const std::set<TA> &s0, const std::set<TA> &s1)
	{
		Communicate_Map_Period::Judge_Map2_First<TA> judge;
		judge.s0 = s0;
		judge.s1 = s1;
		return Communicate_Tensors_Map::comm_map2(mpi_comm, Ds_in, judge);
	}

	template<typename TA, typename TAC, typename Tdata>
	std::map<TA,std::map<TAC,std::map<TAC,Tensor<Tdata>>>>
	comm_map3_first(
		const MPI_Comm &mpi_comm,
		const std::map<TA,std::map<TAC,std::map<TAC,Tensor<Tdata>>>> &Ds_in,
		const std::set<TA> &s0, const std::set<TA> &s1, const std::set<TA> &s2)
	{
		Communicate_Map_Period::Judge_Map3_First<TA> judge;
		judge.s0 = s0;
		judge.s1 = s1;
		judge.s2 = s2;
		return Communicate_Tensors_Map::comm_map3(mpi_comm, Ds_in, judge);
	}

	template<typename TA, typename TC, typename Tdata>
	std::map<TA,std::map<std::pair<TA,TC>,Tensor<Tdata>>>
	comm_map2_period(
		const MPI_Comm &mpi_comm,
		const std::map<TA,std::map<std::pair<TA,TC>,Tensor<Tdata>>> &Ds_in,
		const std::set<std::pair<TA,TC>> &s0, const std::set<std::pair<TA,TC>> &s1,
		const TC &period)
	{
		Communicate_Map_Period::Judge_Map2_Period<TA,TC> judge;
		judge.period = period;
		judge.s0 = s0;
		judge.s1 = s1;
		return Communicate_Tensors_Map::comm_map2(mpi_comm, Ds_in, judge);
	}

	template<typename TA, typename TC, typename Tdata>
	std::map<TA,std::map<std::pair<TA,TC>,std::map<std::pair<TA,TC>,Tensor<Tdata>>>>
	comm_map3_period(
		const MPI_Comm &mpi_comm,
		const std::map<TA,std::map<std::pair<TA,TC>,std::map<std::pair<TA,TC>,Tensor<Tdata>>>> &Ds_in,
		const std::set<std::pair<TA,TC>> &s0, const std::set<std::pair<TA,TC>> &s1, const std::set<std::pair<TA,TC>> &s2,
		const TC &period)
	{
		Communicate_Map_Period::Judge_Map3_Period<TA,TC> judge;
		judge.period = period;
		judge.s0 = s0;
		judge.s1 = s1;
		judge.s2 = s2;
		return Communicate_Tensors_Map::comm_map3(mpi_comm, Ds_in, judge);
	}
}