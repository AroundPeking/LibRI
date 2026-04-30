// ===================
//  Author: Peize Lin
//  date: 2023.08.02
// ===================

#pragma once

#include "LRI.h"
#include "LRI_Cal_Aux.h"
#include "../global/Array_Operator.h"
#include "../global/Tensor_Multiply.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <omp.h>
#ifdef __MKL_RI
#include <mkl_service.h>
#endif

namespace RI
{

struct LRI_Weighted_Short_Screen_Config
{
	double threshold = -1.0;
	bool stats_only = false;
	bool enabled = false;
};

struct LRI_Weighted_Short_Screen_Stats
{
	std::uint64_t candidates = 0;
	std::uint64_t skips = 0;
	double max_score = 0.0;
};

namespace LRI_Weighted_Short_Screen_Detail
{
	inline std::mutex &get_mutex()
	{
		static std::mutex mtx;
		return mtx;
	}
	inline std::unordered_map<const void*, LRI_Weighted_Short_Screen_Config> &get_configs()
	{
		static std::unordered_map<const void*, LRI_Weighted_Short_Screen_Config> configs;
		return configs;
	}
	inline std::unordered_map<const void*, LRI_Weighted_Short_Screen_Stats> &get_stats()
	{
		static std::unordered_map<const void*, LRI_Weighted_Short_Screen_Stats> stats;
		return stats;
	}
}

inline void lri_set_weighted_short_screen_config(
	const void *lri_key,
	const LRI_Weighted_Short_Screen_Config &config)
{
	std::lock_guard<std::mutex> lock(LRI_Weighted_Short_Screen_Detail::get_mutex());
	LRI_Weighted_Short_Screen_Detail::get_configs()[lri_key] = config;
	LRI_Weighted_Short_Screen_Detail::get_stats()[lri_key] = LRI_Weighted_Short_Screen_Stats{};
}

inline LRI_Weighted_Short_Screen_Config lri_get_weighted_short_screen_config(const void *lri_key)
{
	std::lock_guard<std::mutex> lock(LRI_Weighted_Short_Screen_Detail::get_mutex());
	const auto it = LRI_Weighted_Short_Screen_Detail::get_configs().find(lri_key);
	return (it!=LRI_Weighted_Short_Screen_Detail::get_configs().end())
		? it->second
		: LRI_Weighted_Short_Screen_Config{};
}

inline void lri_set_weighted_short_screen_stats(
	const void *lri_key,
	const LRI_Weighted_Short_Screen_Stats &stats)
{
	std::lock_guard<std::mutex> lock(LRI_Weighted_Short_Screen_Detail::get_mutex());
	LRI_Weighted_Short_Screen_Detail::get_stats()[lri_key] = stats;
}

inline LRI_Weighted_Short_Screen_Stats lri_get_weighted_short_screen_stats(const void *lri_key)
{
	std::lock_guard<std::mutex> lock(LRI_Weighted_Short_Screen_Detail::get_mutex());
	const auto it = LRI_Weighted_Short_Screen_Detail::get_stats().find(lri_key);
	return (it!=LRI_Weighted_Short_Screen_Detail::get_stats().end())
		? it->second
		: LRI_Weighted_Short_Screen_Stats{};
}

inline bool lri_weighted_short_is_short_label(const Label::ab_ab label)
{
	switch(label)
	{
		case Label::ab_ab::a0b0_a1b1:
		case Label::ab_ab::a0b0_a1b2:
		case Label::ab_ab::a0b0_a2b1:
		case Label::ab_ab::a0b0_a2b2:
			return true;
		default:
			return false;
	}
}

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void LRI<TA,Tcell,Ndim,Tdata>::cal_loop3(
	const std::vector<Label::ab_ab> &labels,
	std::map<TA, std::map<TAC, Tensor<Tdata>>> &Ds_result,
	const double fac_add_Ds)
{
	using namespace Array_Operator;

	const Data_Pack_Wrapper<TA,TC,Tdata> data_wrapper(this->data_pool, this->data_ab_name);
	const LRI_Cal_Tools<TA,TC,Tdata> tools(this->period, this->data_pool, this->data_ab_name);

	std::map<TA, std::map<TAC, Tensor<Tdata>>> Ds_a_transpose, Ds_b_transpose;
	std::tie(Ds_a_transpose, Ds_b_transpose) = tools.cal_Ds_transpose(labels);

	const void *weighted_short_lri_key = static_cast<const void*>(this);
	const LRI_Weighted_Short_Screen_Config weighted_short_config
		= lri_get_weighted_short_screen_config(weighted_short_lri_key);
	const bool weighted_short_enabled
		= weighted_short_config.enabled && (weighted_short_config.threshold > 0.0);
	std::uint64_t weighted_short_candidates = 0;
	std::uint64_t weighted_short_skips = 0;
	double weighted_short_max_score = 0.0;

  #ifdef __MKL_RI
	const std::size_t mkl_threads = mkl_get_max_threads();
	mkl_set_num_threads(1);
  #endif

	std::map<TA, omp_lock_t> lock_Ds_result_add_map = LRI_Cal_Aux::init_lock_result(labels, this->parallel->list_A, Ds_result);

	#pragma omp parallel
	{
		std::map<TA, std::map<TAC, Tensor<Tdata>>> Ds_result_thread;
		std::unordered_map<const Tensor<Tdata>*,double> weighted_short_block_max_cache;
		std::uint64_t weighted_short_candidates_thread = 0;
		std::uint64_t weighted_short_skips_thread = 0;
		double weighted_short_max_score_thread = 0.0;
		const auto weighted_short_block_max_cached =
			[&weighted_short_block_max_cache](const Tensor<Tdata> &D) -> double
			{
				const auto it = weighted_short_block_max_cache.find(&D);
				if(it != weighted_short_block_max_cache.end())
					return it->second;
				const double value = D.norm(std::numeric_limits<double>::max());
				weighted_short_block_max_cache[&D] = value;
				return value;
			};
		const auto weighted_short_path_below_threshold =
			[&weighted_short_candidates_thread, &weighted_short_skips_thread, &weighted_short_max_score_thread,
			 &weighted_short_config, weighted_short_enabled](const double score) -> bool
			{
				if(!weighted_short_enabled)
					return false;
				++weighted_short_candidates_thread;
				weighted_short_max_score_thread = std::max(weighted_short_max_score_thread, score);
				const bool below_threshold = (score < weighted_short_config.threshold);
				if(below_threshold)
					++weighted_short_skips_thread;
				return below_threshold;
			};

		for(const Label::ab_ab &label : labels)
		{
			const bool weighted_short_enable_label = weighted_short_enabled && lri_weighted_short_is_short_label(label);
			const std::vector<TA>  list_Aa01_Da = LRI_Cal_Aux::filter_list_map( this->parallel->list_A.at(Label_Tools::to_Aab_Aab(label)).a01, data_wrapper(Label::ab::a).Ds_ab );
			const std::vector<TAC> list_Ab01_Db = LRI_Cal_Aux::filter_list_map( this->parallel->list_A.at(Label_Tools::to_Aab_Aab(label)).b01, data_wrapper(Label::ab::b).Ds_ab );
			const std::vector<TAC> list_Aa2_Da  = LRI_Cal_Aux::filter_list_set( this->parallel->list_A.at(Label_Tools::to_Aab_Aab(label)).a2,  data_wrapper(Label::ab::a).index_Ds_ab[0] );
			const std::vector<TAC> list_Ab2_Db  = LRI_Cal_Aux::filter_list_set( this->parallel->list_A.at(Label_Tools::to_Aab_Aab(label)).b2,  data_wrapper(Label::ab::b).index_Ds_ab[0] );
			switch(label)
			{

			  // Aab_Aab::a01b01_a01b01

				case Label::ab_ab::a0b0_a1b1:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map( LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a0b0).Ds_ab ),
						data_wrapper(Label::ab::a1b1).Ds_ab );
					const std::vector<TAC> &list_Aa2 =
						list_Aa2_Da;
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set( LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a0b0).index_Ds_ab[0]),
						data_wrapper(Label::ab::a1b1).index_Ds_ab[0]);
					const std::vector<TAC> &list_Ab2 =
						list_Ab2_Db;

					for(const TAC &Aa2 : list_Aa2)
					{
						if(this->filter_atom->filter_for1(label,Aa2))	continue;
						std::map<TAC,Tensor<Tdata>> Ds_result_fixed;

						#pragma omp for schedule(dynamic) nowait
						for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
						{
							const TAC &Ab01 = list_Ab01[ib01];
							if(this->filter_atom->filter_for2(label,Aa2,Ab01))	continue;
							// D_mul = D_a * D_a0b0 * D_a1b1
							Tensor<Tdata> D_mul;
							std::vector<char> weighted_short_keep_ab2;
							if(weighted_short_enable_label)
								weighted_short_keep_ab2.assign(list_Ab2.size(), 0);
							for(const TA &Aa01 : list_Aa01)
							{
								if(this->filter_atom->filter_for31(label,Aa2,Ab01,Aa01))	continue;
								const Tensor<Tdata> &D_a = tools.get_Ds_ab(Label::ab::a, Aa01, Aa2);
								if(D_a.empty())	continue;
								const Tensor<Tdata> &D_a0b0 = tools.get_Ds_ab(Label::ab::a0b0, Aa01, Ab01);
								if(D_a0b0.empty())	continue;
								const Tensor<Tdata> &D_a1b1 = tools.get_Ds_ab(Label::ab::a1b1, Aa01, Ab01);
								if(D_a1b1.empty())	continue;
								bool weighted_short_has_kept_ab2 = true;
								if(weighted_short_enable_label)
								{
									weighted_short_has_kept_ab2 = false;
									const double weighted_short_left_score
										= weighted_short_block_max_cached(D_a)
										* weighted_short_block_max_cached(D_a0b0)
										* weighted_short_block_max_cached(D_a1b1);
									for(std::size_t ib2=0; ib2<list_Ab2.size(); ++ib2)
									{
										const TAC &Ab2 = list_Ab2[ib2];
										if(this->filter_atom->filter_for32(label,Aa2,Ab01,Ab2))	continue;
										const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
										if(D_b.empty())	continue;
										const double weighted_short_contraction_factor
											= static_cast<double>(D_a.shape[0])
											* static_cast<double>(D_a.shape[1])
											* static_cast<double>(D_b.shape[0])
											* static_cast<double>(D_b.shape[1]);
										const bool weighted_short_below
											= weighted_short_path_below_threshold(
												weighted_short_contraction_factor
												* weighted_short_left_score
												* weighted_short_block_max_cached(D_b));
										const bool weighted_short_keep
											= !weighted_short_below || weighted_short_config.stats_only;
										weighted_short_keep_ab2[ib2] = weighted_short_keep_ab2[ib2] || (weighted_short_keep ? 1 : 0);
										weighted_short_has_kept_ab2 = weighted_short_has_kept_ab2 || weighted_short_keep;
									}
									if(!weighted_short_has_kept_ab2 && !weighted_short_config.stats_only)	continue;
								}

								// a1a2b0 = a0a1a2 * a0b0
								const Tensor<Tdata> D_tmp1 = Tensor_Multiply::x1x2y1_ax1x2_ay1(D_a, D_a0b0);
								// a2b0b1 = a1a2b0 * a1b1
								Tensor<Tdata> D_tmp2 = Tensor_Multiply::x1x2y1_ax1x2_ay1(D_tmp1, D_a1b1);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp2), D_mul);
							}
							if(D_mul.empty())	continue;

							// D_result = D_mul * D_b
							for(std::size_t ib2=0; ib2<list_Ab2.size(); ++ib2)
							{
								const TAC &Ab2 = list_Ab2[ib2];
								if(this->filter_atom->filter_for32(label,Aa2,Ab01,Ab2))	continue;
								const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
								if(D_b.empty())	continue;
								if(weighted_short_enable_label && !weighted_short_keep_ab2[ib2])	continue;
								// a2b2 = a2b0b1 * b0b1b2
								Tensor<Tdata> D_tmp3 = Tensor_Multiply::x0y2_x0ab_aby2(D_mul, D_b);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp3), Ds_result_fixed[Ab2]);
							}
						} // end for Ab01

						if(!Ds_result_fixed.empty())
							LRI_Cal_Aux::add_Ds(LRI_Cal_Aux::Ds_translate(std::move(Ds_result_fixed), Aa2.second, this->period),
												Ds_result_thread[Aa2.first]);
						LRI_Cal_Aux::add_Ds_omp_try_map(Ds_result_thread, Ds_result, lock_Ds_result_add_map, fac_add_Ds);
					} // end for Ab2
				} break; // end case a0b0_a1b1

				case Label::ab_ab::a0b1_a1b0:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map( LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a0b1).Ds_ab ),
						data_wrapper(Label::ab::a1b0).Ds_ab );
					const std::vector<TAC> &list_Aa2 =
						list_Aa2_Da;
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set( LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a0b1).index_Ds_ab[0]),
						data_wrapper(Label::ab::a1b0).index_Ds_ab[0]);
					const std::vector<TAC> &list_Ab2 =
						list_Ab2_Db;

					for(const TAC &Aa2 : list_Aa2)
					{
						if(this->filter_atom->filter_for1(label,Aa2))	continue;
						std::map<TAC,Tensor<Tdata>> Ds_result_fixed;

						#pragma omp for schedule(dynamic) nowait
						for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
						{
							const TAC &Ab01 = list_Ab01[ib01];
							if(this->filter_atom->filter_for2(label,Aa2,Ab01))	continue;
							// D_mul = D_a * D_a0b1 * D_a1b0
							Tensor<Tdata> D_mul;
							for(const TA &Aa01 : list_Aa01)
							{
								if(this->filter_atom->filter_for31(label,Aa2,Ab01,Aa01))	continue;
								const Tensor<Tdata> &D_a = Global_Func::find(Ds_a_transpose, Aa01, Aa2);
								if(D_a.empty())	continue;
								const Tensor<Tdata> &D_a0b1 = tools.get_Ds_ab(Label::ab::a0b1, Aa01, Ab01);
								if(D_a0b1.empty())	continue;
								const Tensor<Tdata> &D_a1b0 = tools.get_Ds_ab(Label::ab::a1b0, Aa01, Ab01);
								if(D_a1b0.empty())	continue;

								// a0a2b0 = a1a0a2 * a1b0
								const Tensor<Tdata> D_tmp1 = Tensor_Multiply::x1x2y1_ax1x2_ay1(D_a, D_a1b0);
								// a2b0b1 = a0a2b0 * a0b1
								Tensor<Tdata> D_tmp2 = Tensor_Multiply::x1x2y1_ax1x2_ay1(D_tmp1, D_a0b1);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp2), D_mul);
							}
							if(D_mul.empty())	continue;

							// D_result = D_mul * D_b
							for(const TAC &Ab2 : list_Ab2)
							{
								if(this->filter_atom->filter_for32(label,Aa2,Ab01,Ab2))	continue;
								const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
								if(D_b.empty())	continue;
								// a2b2 = a2b0b1 * b0b1b2
								Tensor<Tdata> D_tmp3 = Tensor_Multiply::x0y2_x0ab_aby2(D_mul, D_b);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp3), Ds_result_fixed[Ab2]);
							}
						} // end for Ab01

						if(!Ds_result_fixed.empty())
							LRI_Cal_Aux::add_Ds(LRI_Cal_Aux::Ds_translate(std::move(Ds_result_fixed), Aa2.second, this->period),
												Ds_result_thread[Aa2.first]);
						LRI_Cal_Aux::add_Ds_omp_try_map(Ds_result_thread, Ds_result, lock_Ds_result_add_map, fac_add_Ds);
					} // end for Ab2
				} break; // end case a0b1_a1b0

			  // Aab_Aab::a01b01_a01b2

				case Label::ab_ab::a0b0_a1b2:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map( LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a0b0).Ds_ab ),
						data_wrapper(Label::ab::a1b2).Ds_ab );
					const std::vector<TAC> &list_Aa2 =
						list_Aa2_Da;
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a0b0).index_Ds_ab[0]);
					const std::vector<TAC>  list_Ab2 = LRI_Cal_Aux::filter_list_set(
						list_Ab2_Db,
						data_wrapper(Label::ab::a1b2).index_Ds_ab[0]);

					for(const TAC &Ab01 : list_Ab01)
					{
						if(this->filter_atom->filter_for1(label,Ab01))	continue;
						std::map<TAC,Tensor<Tdata>> Ds_result_fixed;

						#pragma omp for schedule(dynamic) nowait
						for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01)
						{
							const TA &Aa01 = list_Aa01[ia01];
							if(this->filter_atom->filter_for2(label,Ab01,Aa01))	continue;
							const Tensor<Tdata> &D_a0b0 = tools.get_Ds_ab(Label::ab::a0b0, Aa01, Ab01);
							if(D_a0b0.empty())	continue;
							// D_mul = D_b * D_a1b2
							Tensor<Tdata> D_mul;
							std::vector<char> weighted_short_keep_Aa2;
							if(weighted_short_enable_label)
								weighted_short_keep_Aa2.assign(list_Aa2.size(), 0);
							for(const TAC &Ab2 : list_Ab2)
							{
								if(this->filter_atom->filter_for31(label,Ab01,Aa01,Ab2))	continue;
								const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
								if(D_b.empty())	continue;
								const Tensor<Tdata> &D_a1b2 = tools.get_Ds_ab(Label::ab::a1b2, Aa01, Ab2);
								if(D_a1b2.empty())	continue;
								bool weighted_short_keep_ab2 = true;
								if(weighted_short_enable_label)
								{
									weighted_short_keep_ab2 = false;
									const double weighted_short_left_score
										= weighted_short_block_max_cached(D_b)
										* weighted_short_block_max_cached(D_a1b2)
										* weighted_short_block_max_cached(D_a0b0);
									for(std::size_t ia2=0; ia2<list_Aa2.size(); ++ia2)
									{
										const TAC &Aa2 = list_Aa2[ia2];
										if(this->filter_atom->filter_for32(label,Ab01,Aa01,Aa2))	continue;
										const Tensor<Tdata> &D_a = Global_Func::find(Ds_a_transpose, Aa01, Aa2);
										if(D_a.empty())	continue;
										const double weighted_short_contraction_factor
											= static_cast<double>(D_b.shape[2])
											* static_cast<double>(D_b.shape[0])
											* static_cast<double>(D_a.shape[0])
											* static_cast<double>(D_a.shape[1]);
										const bool weighted_short_below
											= weighted_short_path_below_threshold(
												weighted_short_contraction_factor
												* weighted_short_left_score
												* weighted_short_block_max_cached(D_a));
										const bool weighted_short_keep
											= !weighted_short_below || weighted_short_config.stats_only;
										weighted_short_keep_Aa2[ia2] = weighted_short_keep_Aa2[ia2] || (weighted_short_keep ? 1 : 0);
										weighted_short_keep_ab2 = weighted_short_keep_ab2 || weighted_short_keep;
									}
									if(!weighted_short_keep_ab2 && !weighted_short_config.stats_only)	continue;
								}

								// b0b1a1 = b0b1b2 * a1b2
								Tensor<Tdata> D_tmp1 = Tensor_Multiply::x0x1y0_x0x1a_y0a(D_b, D_a1b2);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp1), D_mul);
							}
							if(D_mul.empty())	continue;

							// D_result = D_mul * D_a * D_a0b0
							for(std::size_t ia2=0; ia2<list_Aa2.size(); ++ia2)
							{
								const TAC &Aa2 = list_Aa2[ia2];
								if(this->filter_atom->filter_for32(label,Ab01,Aa01,Aa2))	continue;
								const Tensor<Tdata> &D_a = Global_Func::find(Ds_a_transpose, Aa01, Aa2);
								if(D_a.empty())	continue;
								if(weighted_short_enable_label && !weighted_short_keep_Aa2[ia2])	continue;
								// b1a1a0 = b0b1a1 * a0b0
								const Tensor<Tdata> D_tmp2 = Tensor_Multiply::x1x2y0_ax1x2_y0a(D_mul, D_a0b0);
								// a2b1 = a1a0a2 * b1a1a0
								Tensor<Tdata> D_tmp3 = Tensor_Multiply::x2y0_abx2_y0ab(D_a, D_tmp2);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp3), Ds_result_fixed[Aa2]);
							}
						} // end for Aa01

						if(!Ds_result_fixed.empty())
							LRI_Cal_Aux::add_Ds(LRI_Cal_Aux::Ds_exchange(std::move(Ds_result_fixed), Ab01, this->period),
												Ds_result_thread);
						LRI_Cal_Aux::add_Ds_omp_try_map(Ds_result_thread, Ds_result, lock_Ds_result_add_map, fac_add_Ds);
					} // end for Ab01
				} break; // end case a0b0_a1b2

				case Label::ab_ab::a0b1_a1b2:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map( LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a0b1).Ds_ab ),
						data_wrapper(Label::ab::a1b2).Ds_ab );
					const std::vector<TAC> &list_Aa2 =
						list_Aa2_Da;
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a0b1).index_Ds_ab[0]);
					const std::vector<TAC>  list_Ab2 = LRI_Cal_Aux::filter_list_set(
						list_Ab2_Db,
						data_wrapper(Label::ab::a1b2).index_Ds_ab[0]);

					for(const TAC &Ab01 : list_Ab01)
					{
						if(this->filter_atom->filter_for1(label,Ab01))	continue;
						std::map<TAC,Tensor<Tdata>> Ds_result_fixed;

						#pragma omp for schedule(dynamic) nowait
						for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01)
						{
							const TA &Aa01 = list_Aa01[ia01];
							if(this->filter_atom->filter_for2(label,Ab01,Aa01))	continue;
							const Tensor<Tdata> &D_a0b1 = tools.get_Ds_ab(Label::ab::a0b1, Aa01, Ab01);
							if(D_a0b1.empty())	continue;
							// D_mul = D_b * D_a1b2
							Tensor<Tdata> D_mul;
							for(const TAC &Ab2 : list_Ab2)
							{
								if(this->filter_atom->filter_for31(label,Ab01,Aa01,Ab2))	continue;
								const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
								if(D_b.empty())	continue;
								const Tensor<Tdata> &D_a1b2 = tools.get_Ds_ab(Label::ab::a1b2, Aa01, Ab2);
								if(D_a1b2.empty())	continue;

								// a1b0b1 = a1b2 * b0b1b2
								Tensor<Tdata> D_tmp1 = Tensor_Multiply::x0y0y1_x0a_y0y1a(D_a1b2, D_b);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp1), D_mul);
							}
							if(D_mul.empty())	continue;

							// D_result = D_mul * D_a * D_a0b1
							for(const TAC &Aa2 : list_Aa2)
							{
								if(this->filter_atom->filter_for32(label,Ab01,Aa01,Aa2))	continue;
								const Tensor<Tdata> &D_a = tools.get_Ds_ab(Label::ab::a, Aa01, Aa2);
								if(D_a.empty())	continue;
								// a0a1b0 = a0b1 * a1b0b1
								const Tensor<Tdata> D_tmp2 = Tensor_Multiply::x0y0y1_x0a_y0y1a(D_a0b1, D_mul);
								// a2b0 = a0a1a2 * a0a1b0
								Tensor<Tdata> D_tmp3 = Tensor_Multiply::x2y2_abx2_aby2(D_a, D_tmp2);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp3), Ds_result_fixed[Aa2]);
							}
						} // end for Aa01

						if(!Ds_result_fixed.empty())
							LRI_Cal_Aux::add_Ds(LRI_Cal_Aux::Ds_exchange(std::move(Ds_result_fixed), Ab01, this->period),
												Ds_result_thread);
						LRI_Cal_Aux::add_Ds_omp_try_map(Ds_result_thread, Ds_result, lock_Ds_result_add_map, fac_add_Ds);
					} // end for Ab01
				} break; // end case a0b1_a1b2

				case Label::ab_ab::a0b2_a1b0:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map( LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a1b0).Ds_ab ),
						data_wrapper(Label::ab::a0b2).Ds_ab );
					const std::vector<TAC> &list_Aa2 =
						list_Aa2_Da;
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a1b0).index_Ds_ab[0]);
					const std::vector<TAC>  list_Ab2 = LRI_Cal_Aux::filter_list_set(
						list_Ab2_Db,
						data_wrapper(Label::ab::a0b2).index_Ds_ab[0]);

					for(const TAC &Ab01 : list_Ab01)
					{
						if(this->filter_atom->filter_for1(label,Ab01))	continue;
						std::map<TAC,Tensor<Tdata>> Ds_result_fixed;

						#pragma omp for schedule(dynamic) nowait
						for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01)
						{
							const TA &Aa01 = list_Aa01[ia01];
							if(this->filter_atom->filter_for2(label,Ab01,Aa01))	continue;
							const Tensor<Tdata> &D_a1b0 = tools.get_Ds_ab(Label::ab::a1b0, Aa01, Ab01);
							if(D_a1b0.empty())	continue;
							// D_mul = D_b * D_a0b2
							Tensor<Tdata> D_mul;
							for(const TAC &Ab2 : list_Ab2)
							{
								if(this->filter_atom->filter_for31(label,Ab01,Aa01,Ab2))	continue;
								const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
								if(D_b.empty())	continue;
								const Tensor<Tdata> &D_a0b2 = tools.get_Ds_ab(Label::ab::a0b2, Aa01, Ab2);
								if(D_a0b2.empty())	continue;

								// b0b1a0 = b0b1b2 * a0b2
								Tensor<Tdata> D_tmp1 = Tensor_Multiply::x0x1y0_x0x1a_y0a(D_b, D_a0b2);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp1), D_mul);
							}
							if(D_mul.empty())	continue;

							// D_result = D_mul * D_a * D_a1b0
							for(const TAC &Aa2 : list_Aa2)
							{
								if(this->filter_atom->filter_for32(label,Ab01,Aa01,Aa2))	continue;
								const Tensor<Tdata> &D_a = tools.get_Ds_ab(Label::ab::a, Aa01, Aa2);
								if(D_a.empty())	continue;
								// b1a0a1 = b0b1a0 * a1b0
								const Tensor<Tdata> D_tmp2 = Tensor_Multiply::x1x2y0_ax1x2_y0a(D_mul, D_a1b0);
								// a2b1 = a0a1a2 * b1a0a1
								Tensor<Tdata> D_tmp3 = Tensor_Multiply::x2y0_abx2_y0ab(D_a, D_tmp2);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp3), Ds_result_fixed[Aa2]);
							}
						} // end for Aa01

						if(!Ds_result_fixed.empty())
							LRI_Cal_Aux::add_Ds(LRI_Cal_Aux::Ds_exchange(std::move(Ds_result_fixed), Ab01, this->period),
												Ds_result_thread);
						LRI_Cal_Aux::add_Ds_omp_try_map(Ds_result_thread, Ds_result, lock_Ds_result_add_map, fac_add_Ds);
					} // end for Ab01
				} break; // end case a0b2_a1b0

				case Label::ab_ab::a0b2_a1b1:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map( LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a1b1).Ds_ab ),
						data_wrapper(Label::ab::a0b2).Ds_ab );
					const std::vector<TAC> &list_Aa2 =
						list_Aa2_Da;
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a1b1).index_Ds_ab[0]);
					const std::vector<TAC>  list_Ab2 = LRI_Cal_Aux::filter_list_set(
						list_Ab2_Db,
						data_wrapper(Label::ab::a0b2).index_Ds_ab[0]);

					for(const TAC &Ab01 : list_Ab01)
					{
						if(this->filter_atom->filter_for1(label,Ab01))	continue;
						std::map<TAC,Tensor<Tdata>> Ds_result_fixed;

						#pragma omp for schedule(dynamic) nowait
						for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01)
						{
							const TA &Aa01 = list_Aa01[ia01];
							if(this->filter_atom->filter_for2(label,Ab01,Aa01))	continue;
							const Tensor<Tdata> &D_a1b1 = tools.get_Ds_ab(Label::ab::a1b1, Aa01, Ab01);
							if(D_a1b1.empty())	continue;
							// D_mul = D_b * D_a0b2
							Tensor<Tdata> D_mul;
							for(const TAC &Ab2 : list_Ab2)
							{
								if(this->filter_atom->filter_for31(label,Ab01,Aa01,Ab2))	continue;
								const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
								if(D_b.empty())	continue;
								const Tensor<Tdata> &D_a0b2 = tools.get_Ds_ab(Label::ab::a0b2, Aa01, Ab2);
								if(D_a0b2.empty())	continue;

								// a0b0b1 = a0b2 * b0b1b2
								Tensor<Tdata> D_tmp1 = Tensor_Multiply::x0y0y1_x0a_y0y1a(D_a0b2, D_b);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp1), D_mul);
							}
							if(D_mul.empty())	continue;

							// D_result = D_mul * D_a * D_a1b1
							for(const TAC &Aa2 : list_Aa2)
							{
								if(this->filter_atom->filter_for32(label,Ab01,Aa01,Aa2))	continue;
								const Tensor<Tdata> &D_a = Global_Func::find(Ds_a_transpose, Aa01, Aa2);
								if(D_a.empty())	continue;
								// a1a0b0 = a1b1 * a0b0b1
								const Tensor<Tdata> D_tmp2 = Tensor_Multiply::x0y0y1_x0a_y0y1a(D_a1b1, D_mul);
								// a2b0 = a1a0a2 * a1a0b0
								Tensor<Tdata> D_tmp3 = Tensor_Multiply::x2y2_abx2_aby2(D_a, D_tmp2);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp3), Ds_result_fixed[Aa2]);
							}
						} // end for Aa01

						if(!Ds_result_fixed.empty())
							LRI_Cal_Aux::add_Ds(LRI_Cal_Aux::Ds_exchange(std::move(Ds_result_fixed), Ab01, this->period),
												Ds_result_thread);
						LRI_Cal_Aux::add_Ds_omp_try_map(Ds_result_thread, Ds_result, lock_Ds_result_add_map, fac_add_Ds);
					} // end for Ab01
				} break; // end case a0b2_a1b1

			  // Aab_Aab::a01b01_a2b01

				case Label::ab_ab::a0b0_a2b1:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a0b0).Ds_ab );
					const std::vector<TAC>  list_Aa2 = LRI_Cal_Aux::filter_list_map(
						list_Aa2_Da,
						data_wrapper(Label::ab::a2b1).Ds_ab );
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set( LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a0b0).index_Ds_ab[0]),
						data_wrapper(Label::ab::a2b1).index_Ds_ab[0]);
					const std::vector<TAC> &list_Ab2 =
						list_Ab2_Db;

					for(const TA &Aa01 : list_Aa01)
					{
						if(this->filter_atom->filter_for1(label,Aa01))	continue;
						std::map<TAC,Tensor<Tdata>> Ds_result_fixed;

						#pragma omp for schedule(dynamic) nowait
						for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
						{
							const TAC &Ab01 = list_Ab01[ib01];
							if(this->filter_atom->filter_for2(label,Aa01,Ab01))	continue;
							const Tensor<Tdata> &D_a0b0 = tools.get_Ds_ab(Label::ab::a0b0, Aa01, Ab01);
							if(D_a0b0.empty())	continue;
							// D_mul = D_a * D_a2b1
							Tensor<Tdata> D_mul;
							std::vector<char> weighted_short_keep_Ab2;
							if(weighted_short_enable_label)
							{
								weighted_short_keep_Ab2.assign(list_Ab2.size(), 0);
							}
							for(const TAC &Aa2 : list_Aa2)
							{
								if(this->filter_atom->filter_for31(label,Aa01,Ab01,Aa2))	continue;
								const Tensor<Tdata> &D_a = Global_Func::find(Ds_a_transpose, Aa01, Aa2);
								if(D_a.empty())	continue;
								const Tensor<Tdata> &D_a2b1 = tools.get_Ds_ab(Label::ab::a2b1, Aa2, Ab01);
								if(D_a2b1.empty())	continue;
								bool weighted_short_keep_a2 = true;
								if(weighted_short_enable_label)
								{
									weighted_short_keep_a2 = false;
									const double weighted_short_left_score
										= weighted_short_block_max_cached(D_a2b1)
										* weighted_short_block_max_cached(D_a)
										* weighted_short_block_max_cached(D_a0b0);
									for(std::size_t ib2=0; ib2<list_Ab2.size(); ++ib2)
									{
										const TAC &Ab2 = list_Ab2[ib2];
										if(this->filter_atom->filter_for32(label,Aa01,Ab01,Ab2))	continue;
										const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
										if(D_b.empty())	continue;
										const double weighted_short_contraction_factor
											= static_cast<double>(D_a2b1.shape[0])
											* static_cast<double>(D_a0b0.shape[0])
											* static_cast<double>(D_b.shape[0])
											* static_cast<double>(D_b.shape[1]);
										const bool weighted_short_below
											= weighted_short_path_below_threshold(
												weighted_short_contraction_factor
												* weighted_short_left_score
												* weighted_short_block_max_cached(D_b));
										const bool weighted_short_keep
											= !weighted_short_below || weighted_short_config.stats_only;
										weighted_short_keep_Ab2[ib2] = weighted_short_keep_Ab2[ib2] || (weighted_short_keep ? 1 : 0);
										weighted_short_keep_a2 = weighted_short_keep_a2 || weighted_short_keep;
									}
									if(!weighted_short_keep_a2 && !weighted_short_config.stats_only)	continue;
								}

								// b1a1a0 = a2b1 * a1a0a2
								Tensor<Tdata> D_tmp1 = Tensor_Multiply::x1y0y1_ax1_y0y1a(D_a2b1, D_a);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp1), D_mul);
							}
							if(D_mul.empty())	continue;

							// D_result = D_mul * D_a0b0 * D_b
							for(std::size_t ib2=0; ib2<list_Ab2.size(); ++ib2)
							{
								const TAC &Ab2 = list_Ab2[ib2];
								if(this->filter_atom->filter_for32(label,Aa01,Ab01,Ab2))	continue;
								const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
								if(D_b.empty())	continue;
								if(weighted_short_enable_label && !weighted_short_keep_Ab2[ib2])	continue;

								// b0b1a1 = a0b0 * b1a1a0
								const Tensor<Tdata> D_tmp2 = Tensor_Multiply::x1y0y1_ax1_y0y1a(D_a0b0, D_mul);
								// a1b2 = b0b1a1 * b0b1b2
								Tensor<Tdata> D_tmp3 = Tensor_Multiply::x2y2_abx2_aby2(D_tmp2, D_b);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp3), Ds_result_fixed[Ab2]);
							}
						} // end for Ab01

						if(!Ds_result_fixed.empty())
							LRI_Cal_Aux::add_Ds(std::move(Ds_result_fixed),
												Ds_result_thread[Aa01]);
						LRI_Cal_Aux::add_Ds_omp_try_map(Ds_result_thread, Ds_result, lock_Ds_result_add_map, fac_add_Ds);
					} // end for Aa01
				} break; // end case a0b0_a2b1

				case Label::ab_ab::a0b1_a2b0:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a0b1).Ds_ab );
					const std::vector<TAC>  list_Aa2 = LRI_Cal_Aux::filter_list_map(
						list_Aa2_Da,
						data_wrapper(Label::ab::a2b0).Ds_ab );
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set( LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a0b1).index_Ds_ab[0]),
						data_wrapper(Label::ab::a2b0).index_Ds_ab[0]);
					const std::vector<TAC> &list_Ab2 =
						list_Ab2_Db;

					for(const TA &Aa01 : list_Aa01)
					{
						if(this->filter_atom->filter_for1(label,Aa01))	continue;
						std::map<TAC,Tensor<Tdata>> Ds_result_fixed;

						#pragma omp for schedule(dynamic) nowait
						for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
						{
							const TAC &Ab01 = list_Ab01[ib01];
							if(this->filter_atom->filter_for2(label,Aa01,Ab01))	continue;
							const Tensor<Tdata> &D_a0b1 = tools.get_Ds_ab(Label::ab::a0b1, Aa01, Ab01);
							if(D_a0b1.empty())	continue;
							// D_mul = D_a * D_a2b0
							Tensor<Tdata> D_mul;
							for(const TAC &Aa2 : list_Aa2)
							{
								if(this->filter_atom->filter_for31(label,Aa01,Ab01,Aa2))	continue;
								const Tensor<Tdata> &D_a = tools.get_Ds_ab(Label::ab::a, Aa01, Aa2);
								if(D_a.empty())	continue;
								const Tensor<Tdata> &D_a2b0 = tools.get_Ds_ab(Label::ab::a2b0, Aa2, Ab01);
								if(D_a2b0.empty())	continue;

								// a0a1b0 = a0a1a2 * a2b0
								Tensor<Tdata> D_tmp1 = Tensor_Multiply::x0x1y1_x0x1a_ay1(D_a, D_a2b0);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp1), D_mul);
							}
							if(D_mul.empty())	continue;

							// D_result = D_mul * D_a0b1 * D_b
							for(const TAC &Ab2 : list_Ab2)
							{
								if(this->filter_atom->filter_for32(label,Aa01,Ab01,Ab2))	continue;
								const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
								if(D_b.empty())	continue;

								// a1b0b1 = a0a1b0 * a0b1
								const Tensor<Tdata> D_tmp2 = Tensor_Multiply::x1x2y1_ax1x2_ay1(D_mul, D_a0b1);
								// a1b2 = a1b0b1 * b0b1b2
								Tensor<Tdata> D_tmp3 = Tensor_Multiply::x0y2_x0ab_aby2(D_tmp2, D_b);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp3), Ds_result_fixed[Ab2]);
							}
						} // end for Ab01

						if(!Ds_result_fixed.empty())
							LRI_Cal_Aux::add_Ds(std::move(Ds_result_fixed),
												Ds_result_thread[Aa01]);
						LRI_Cal_Aux::add_Ds_omp_try_map(Ds_result_thread, Ds_result, lock_Ds_result_add_map, fac_add_Ds);
					} // end for Aa01
				} break; // end case a0b1_a2b0

				case Label::ab_ab::a1b0_a2b1:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a1b0).Ds_ab );
					const std::vector<TAC>  list_Aa2 = LRI_Cal_Aux::filter_list_map(
						list_Aa2_Da,
						data_wrapper(Label::ab::a2b1).Ds_ab );
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set( LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a1b0).index_Ds_ab[0]),
						data_wrapper(Label::ab::a2b1).index_Ds_ab[0]);
					const std::vector<TAC> &list_Ab2 =
						list_Ab2_Db;

					for(const TA &Aa01 : list_Aa01)
					{
						if(this->filter_atom->filter_for1(label,Aa01))	continue;
						std::map<TAC,Tensor<Tdata>> Ds_result_fixed;

						#pragma omp for schedule(dynamic) nowait
						for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
						{
							const TAC &Ab01 = list_Ab01[ib01];
							if(this->filter_atom->filter_for2(label,Aa01,Ab01))	continue;
							const Tensor<Tdata> &D_a1b0 = tools.get_Ds_ab(Label::ab::a1b0, Aa01, Ab01);
							if(D_a1b0.empty())	continue;
							// D_mul = D_a * D_a2b1
							Tensor<Tdata> D_mul;
							for(const TAC &Aa2 : list_Aa2)
							{
								if(this->filter_atom->filter_for31(label,Aa01,Ab01,Aa2))	continue;
								const Tensor<Tdata> &D_a = tools.get_Ds_ab(Label::ab::a, Aa01, Aa2);
								if(D_a.empty())	continue;
								const Tensor<Tdata> &D_a2b1 = tools.get_Ds_ab(Label::ab::a2b1, Aa2, Ab01);
								if(D_a2b1.empty())	continue;

								// b1a0a1 = a2b1 * a0a1a2
								Tensor<Tdata> D_tmp1 = Tensor_Multiply::x1y0y1_ax1_y0y1a(D_a2b1, D_a);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp1), D_mul);
							}
							if(D_mul.empty())	continue;

							// D_result = D_mul * D_a1b0 * D_b
							for(const TAC &Ab2 : list_Ab2)
							{
								if(this->filter_atom->filter_for32(label,Aa01,Ab01,Ab2))	continue;
								const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
								if(D_b.empty())	continue;

								// b0b1a0 = a1b0 * b1a0a1
								const Tensor<Tdata> D_tmp2 = Tensor_Multiply::x1y0y1_ax1_y0y1a(D_a1b0, D_mul);
								// a0b2 = b0b1a0 * b0b1b2
								Tensor<Tdata> D_tmp3 = Tensor_Multiply::x2y2_abx2_aby2(D_tmp2, D_b);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp3), Ds_result_fixed[Ab2]);
							}
						} // end for Ab01

						if(!Ds_result_fixed.empty())
							LRI_Cal_Aux::add_Ds(std::move(Ds_result_fixed),
												Ds_result_thread[Aa01]);
						LRI_Cal_Aux::add_Ds_omp_try_map(Ds_result_thread, Ds_result, lock_Ds_result_add_map, fac_add_Ds);
					} // end for Aa01
				} break; // end case a1b0_a2b1

				case Label::ab_ab::a1b1_a2b0:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a1b1).Ds_ab );
					const std::vector<TAC>  list_Aa2 = LRI_Cal_Aux::filter_list_map(
						list_Aa2_Da,
						data_wrapper(Label::ab::a2b0).Ds_ab );
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set( LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a1b1).index_Ds_ab[0]),
						data_wrapper(Label::ab::a2b0).index_Ds_ab[0]);
					const std::vector<TAC> &list_Ab2 =
						list_Ab2_Db;

					for(const TA &Aa01 : list_Aa01)
					{
						if(this->filter_atom->filter_for1(label,Aa01))	continue;
						std::map<TAC,Tensor<Tdata>> Ds_result_fixed;

						#pragma omp for schedule(dynamic) nowait
						for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
						{
							const TAC &Ab01 = list_Ab01[ib01];
							if(this->filter_atom->filter_for2(label,Aa01,Ab01))	continue;
							const Tensor<Tdata> &D_a1b1 = tools.get_Ds_ab(Label::ab::a1b1, Aa01, Ab01);
							if(D_a1b1.empty())	continue;
							// D_mul = D_a * D_a2b0
							Tensor<Tdata> D_mul;
							for(const TAC &Aa2 : list_Aa2)
							{
								if(this->filter_atom->filter_for31(label,Aa01,Ab01,Aa2))	continue;
								const Tensor<Tdata> &D_a = Global_Func::find(Ds_a_transpose, Aa01, Aa2);
								if(D_a.empty())	continue;
								const Tensor<Tdata> &D_a2b0 = tools.get_Ds_ab(Label::ab::a2b0, Aa2, Ab01);
								if(D_a2b0.empty())	continue;

								// a1a0b0 = a1a0a2 * a2b0
								Tensor<Tdata> D_tmp1 = Tensor_Multiply::x0x1y1_x0x1a_ay1(D_a, D_a2b0);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp1), D_mul);
							}
							if(D_mul.empty())	continue;

							// D_result = D_mul * D_a1b1 * D_b
							for(const TAC &Ab2 : list_Ab2)
							{
								if(this->filter_atom->filter_for32(label,Aa01,Ab01,Ab2))	continue;
								const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
								if(D_b.empty())	continue;

								// a0b0b1 = a1a0b0 * a1b1
								const Tensor<Tdata> D_tmp2 = Tensor_Multiply::x1x2y1_ax1x2_ay1(D_mul, D_a1b1);
								// a0b2 = a0b0b1 * b0b1b2
								Tensor<Tdata> D_tmp3 = Tensor_Multiply::x0y2_x0ab_aby2(D_tmp2, D_b);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp3), Ds_result_fixed[Ab2]);
							}
						} // end for Ab01

						if(!Ds_result_fixed.empty())
							LRI_Cal_Aux::add_Ds(std::move(Ds_result_fixed),
												Ds_result_thread[Aa01]);
						LRI_Cal_Aux::add_Ds_omp_try_map(Ds_result_thread, Ds_result, lock_Ds_result_add_map, fac_add_Ds);
					} // end for Aa01
				} break; // end case a1b1_a2b0

			  // Aab_Aab::a01b01_a2b2

				case Label::ab_ab::a0b0_a2b2:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a0b0).Ds_ab );
					const std::vector<TAC>  list_Aa2 = LRI_Cal_Aux::filter_list_map(
						list_Aa2_Da,
						data_wrapper(Label::ab::a2b2).Ds_ab );
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a0b0).index_Ds_ab[0]);
					const std::vector<TAC>  list_Ab2 = LRI_Cal_Aux::filter_list_set(
						list_Ab2_Db,
						data_wrapper(Label::ab::a2b2).index_Ds_ab[0]);

					for(const TA &Aa01 : list_Aa01)
					{
						if(this->filter_atom->filter_for1(label,Aa01))	continue;
						std::map<TAC,Tensor<Tdata>> Ds_result_fixed;

						#pragma omp for schedule(dynamic) nowait
						for(std::size_t ib2=0; ib2<list_Ab2.size(); ++ib2)
						{
							const TAC &Ab2 = list_Ab2[ib2];
							if(this->filter_atom->filter_for2(label,Aa01,Ab2))	continue;
							// D_mul = D_a * D_a2b2
							Tensor<Tdata> D_mul;
							std::vector<char> weighted_short_keep_Ab01;
							if(weighted_short_enable_label)
							{
								weighted_short_keep_Ab01.assign(list_Ab01.size(), 0);
							}
							for(const TAC &Aa2 : list_Aa2)
							{
								if(this->filter_atom->filter_for31(label,Aa01,Ab2,Aa2))	continue;
								const Tensor<Tdata> &D_a = Global_Func::find(Ds_a_transpose, Aa01, Aa2);
								if(D_a.empty())	continue;
								const Tensor<Tdata> &D_a2b2 = tools.get_Ds_ab(Label::ab::a2b2, Aa2, Ab2);
								if(D_a2b2.empty())	continue;
								bool weighted_short_keep_a2 = true;
								if(weighted_short_enable_label)
								{
									weighted_short_keep_a2 = false;
									const double weighted_short_left_score
										= weighted_short_block_max_cached(D_a2b2)
										* weighted_short_block_max_cached(D_a);
									for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
									{
										const TAC &Ab01 = list_Ab01[ib01];
										if(this->filter_atom->filter_for32(label,Aa01,Ab2,Ab01))	continue;
										const Tensor<Tdata> &D_b = Global_Func::find(
											Ds_b_transpose,
											Ab01.first,
											TAC{Ab2.first, (Ab2.second-Ab01.second)%this->period});
										if(D_b.empty())	continue;
										const Tensor<Tdata> &D_a0b0 = tools.get_Ds_ab(Label::ab::a0b0, Aa01, Ab01);
										if(D_a0b0.empty())	continue;
										const double weighted_short_contraction_factor
											= static_cast<double>(D_a2b2.shape[0])
											* static_cast<double>(D_a0b0.shape[0])
											* static_cast<double>(D_b.shape[1])
											* static_cast<double>(D_b.shape[2]);
										const bool weighted_short_below
											= weighted_short_path_below_threshold(
												weighted_short_contraction_factor
												* weighted_short_left_score
												* weighted_short_block_max_cached(D_a0b0)
												* weighted_short_block_max_cached(D_b));
										const bool weighted_short_keep
											= !weighted_short_below || weighted_short_config.stats_only;
										weighted_short_keep_Ab01[ib01] = weighted_short_keep_Ab01[ib01] || (weighted_short_keep ? 1 : 0);
										weighted_short_keep_a2 = weighted_short_keep_a2 || weighted_short_keep;
									}
									if(!weighted_short_keep_a2 && !weighted_short_config.stats_only)	continue;
								}

								// b2a1a0 = a2b2 * a1a0a2
								Tensor<Tdata> D_tmp1 = Tensor_Multiply::x1y0y1_ax1_y0y1a(D_a2b2, D_a);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp1), D_mul);
							}
							if(D_mul.empty())	continue;

							// D_result = D_mul * D_a0b0 * D_b
							for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
							{
								const TAC &Ab01 = list_Ab01[ib01];
								if(this->filter_atom->filter_for32(label,Aa01,Ab2,Ab01))	continue;
								const Tensor<Tdata> &D_b = Global_Func::find(Ds_b_transpose, Ab01.first, TAC{Ab2.first, (Ab2.second-Ab01.second)%this->period});
								if(D_b.empty())	continue;
								const Tensor<Tdata> &D_a0b0 = tools.get_Ds_ab(Label::ab::a0b0, Aa01, Ab01);
								if(D_a0b0.empty())	continue;
								if(weighted_short_enable_label && !weighted_short_keep_Ab01[ib01])	continue;

								// b0b2a1 = a0b0 * b2a1a0
								const Tensor<Tdata> D_tmp2 = Tensor_Multiply::x1y0y1_ax1_y0y1a(D_a0b0, D_mul);
								// a1b1 = b0b2a1 * b1b0b2
								Tensor<Tdata> D_tmp3 = Tensor_Multiply::x2y0_abx2_y0ab(D_tmp2, D_b);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp3), Ds_result_fixed[Ab01]);
							}
						} // end for Ab01

						if(!Ds_result_fixed.empty())
							LRI_Cal_Aux::add_Ds(std::move(Ds_result_fixed),
												Ds_result_thread[Aa01]);
						LRI_Cal_Aux::add_Ds_omp_try_map(Ds_result_thread, Ds_result, lock_Ds_result_add_map, fac_add_Ds);
					} // end for Aa01
				} break; // end case a0b0_a2b2

				case Label::ab_ab::a0b1_a2b2:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a0b1).Ds_ab );
					const std::vector<TAC>  list_Aa2 = LRI_Cal_Aux::filter_list_map(
						list_Aa2_Da,
						data_wrapper(Label::ab::a2b2).Ds_ab );
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a0b1).index_Ds_ab[0]);
					const std::vector<TAC>  list_Ab2 = LRI_Cal_Aux::filter_list_set(
						list_Ab2_Db,
						data_wrapper(Label::ab::a2b2).index_Ds_ab[0]);

					for(const TA &Aa01 : list_Aa01)
					{
						if(this->filter_atom->filter_for1(label,Aa01))	continue;
						std::map<TAC,Tensor<Tdata>> Ds_result_fixed;

						#pragma omp for schedule(dynamic) nowait
						for(std::size_t ib2=0; ib2<list_Ab2.size(); ++ib2)
						{
							const TAC &Ab2 = list_Ab2[ib2];
							if(this->filter_atom->filter_for2(label,Aa01,Ab2))	continue;
							// D_mul = D_a * D_a2b2
							Tensor<Tdata> D_mul;
							for(const TAC &Aa2 : list_Aa2)
							{
								if(this->filter_atom->filter_for31(label,Aa01,Ab2,Aa2))	continue;
								const Tensor<Tdata> &D_a = Global_Func::find(Ds_a_transpose, Aa01, Aa2);
								if(D_a.empty())	continue;
								const Tensor<Tdata> &D_a2b2 = tools.get_Ds_ab(Label::ab::a2b2, Aa2, Ab2);
								if(D_a2b2.empty())	continue;

								// b2a1a0 = a2b2 * a1a0a2
								Tensor<Tdata> D_tmp1 = Tensor_Multiply::x1y0y1_ax1_y0y1a(D_a2b2, D_a);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp1), D_mul);
							}
							if(D_mul.empty())	continue;

							// D_result = D_mul * D_a0b1 * D_b
							for(const TAC &Ab01 : list_Ab01)
							{
								if(this->filter_atom->filter_for32(label,Aa01,Ab2,Ab01))	continue;
								const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
								if(D_b.empty())	continue;
								const Tensor<Tdata> &D_a0b1 = tools.get_Ds_ab(Label::ab::a0b1, Aa01, Ab01);
								if(D_a0b1.empty())	continue;

								// b1b2a1 = a0b1 * a2a1a0
								const Tensor<Tdata> D_tmp2 = Tensor_Multiply::x1y0y1_ax1_y0y1a(D_a0b1, D_mul);
								// a1b0 = b1b2a1 * b0b1b2
								Tensor<Tdata> D_tmp3 = Tensor_Multiply::x2y0_abx2_y0ab(D_tmp2, D_b);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp3), Ds_result_fixed[Ab01]);
							}
						} // end for Ab01

						if(!Ds_result_fixed.empty())
							LRI_Cal_Aux::add_Ds(std::move(Ds_result_fixed),
												Ds_result_thread[Aa01]);
						LRI_Cal_Aux::add_Ds_omp_try_map(Ds_result_thread, Ds_result, lock_Ds_result_add_map, fac_add_Ds);
					} // end for Aa01
				} break; // end case a0b1_a2b2

				case Label::ab_ab::a1b0_a2b2:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a1b0).Ds_ab );
					const std::vector<TAC>  list_Aa2 = LRI_Cal_Aux::filter_list_map(
						list_Aa2_Da,
						data_wrapper(Label::ab::a2b2).Ds_ab );
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a1b0).index_Ds_ab[0]);
					const std::vector<TAC>  list_Ab2 = LRI_Cal_Aux::filter_list_set(
						list_Ab2_Db,
						data_wrapper(Label::ab::a2b2).index_Ds_ab[0]);

					for(const TA &Aa01 : list_Aa01)
					{
						if(this->filter_atom->filter_for1(label,Aa01))	continue;
						std::map<TAC,Tensor<Tdata>> Ds_result_fixed;

						#pragma omp for schedule(dynamic) nowait
						for(std::size_t ib2=0; ib2<list_Ab2.size(); ++ib2)
						{
							const TAC &Ab2 = list_Ab2[ib2];
							if(this->filter_atom->filter_for2(label,Aa01,Ab2))	continue;
							// D_mul = D_a * D_a2b2
							Tensor<Tdata> D_mul;
							for(const TAC &Aa2 : list_Aa2)
							{
								if(this->filter_atom->filter_for31(label,Aa01,Ab2,Aa2))	continue;
								const Tensor<Tdata> &D_a = tools.get_Ds_ab(Label::ab::a, Aa01, Aa2);
								if(D_a.empty())	continue;
								const Tensor<Tdata> &D_a2b2 = tools.get_Ds_ab(Label::ab::a2b2, Aa2, Ab2);
								if(D_a2b2.empty())	continue;

								// b2a0a1 = a2b2 * a0a1a2
								Tensor<Tdata> D_tmp1 = Tensor_Multiply::x1y0y1_ax1_y0y1a(D_a2b2, D_a);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp1), D_mul);
							}
							if(D_mul.empty())	continue;

							// D_result = D_mul * D_a1b0 * D_b
							for(const TAC &Ab01 : list_Ab01)
							{
								if(this->filter_atom->filter_for32(label,Aa01,Ab2,Ab01))	continue;
								const Tensor<Tdata> &D_b = Global_Func::find(Ds_b_transpose, Ab01.first, TAC{Ab2.first, (Ab2.second-Ab01.second)%this->period});
								if(D_b.empty())	continue;
								const Tensor<Tdata> &D_a1b0 = tools.get_Ds_ab(Label::ab::a1b0, Aa01, Ab01);
								if(D_a1b0.empty())	continue;

								// b0b2a0 = a1b0 * b2a0a1
								const Tensor<Tdata> D_tmp2 = Tensor_Multiply::x1y0y1_ax1_y0y1a(D_a1b0, D_mul);
								// a0b1 = b0b2a0 * b1b0b2
								Tensor<Tdata> D_tmp3 = Tensor_Multiply::x2y0_abx2_y0ab(D_tmp2, D_b);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp3), Ds_result_fixed[Ab01]);
							}
						} // end for Ab01

						if(!Ds_result_fixed.empty())
							LRI_Cal_Aux::add_Ds(std::move(Ds_result_fixed),
												Ds_result_thread[Aa01]);
						LRI_Cal_Aux::add_Ds_omp_try_map(Ds_result_thread, Ds_result, lock_Ds_result_add_map, fac_add_Ds);
					} // end for Aa01
				} break; // end case a1b0_a2b2

				case Label::ab_ab::a1b1_a2b2:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a1b1).Ds_ab );
					const std::vector<TAC>  list_Aa2 = LRI_Cal_Aux::filter_list_map(
						list_Aa2_Da,
						data_wrapper(Label::ab::a2b2).Ds_ab );
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a1b1).index_Ds_ab[0]);
					const std::vector<TAC>  list_Ab2 = LRI_Cal_Aux::filter_list_set(
						list_Ab2_Db,
						data_wrapper(Label::ab::a2b2).index_Ds_ab[0]);

					for(const TA &Aa01 : list_Aa01)
					{
						if(this->filter_atom->filter_for1(label,Aa01))	continue;
						std::map<TAC,Tensor<Tdata>> Ds_result_fixed;

						#pragma omp for schedule(dynamic) nowait
						for(std::size_t ib2=0; ib2<list_Ab2.size(); ++ib2)
						{
							const TAC &Ab2 = list_Ab2[ib2];
							if(this->filter_atom->filter_for2(label,Aa01,Ab2))	continue;
							// D_mul = D_a * D_a2b2
							Tensor<Tdata> D_mul;
							for(const TAC &Aa2 : list_Aa2)
							{
								if(this->filter_atom->filter_for31(label,Aa01,Ab2,Aa2))	continue;
								const Tensor<Tdata> &D_a = tools.get_Ds_ab(Label::ab::a, Aa01, Aa2);
								if(D_a.empty())	continue;
								const Tensor<Tdata> &D_a2b2 = tools.get_Ds_ab(Label::ab::a2b2, Aa2, Ab2);
								if(D_a2b2.empty())	continue;

								// b2a0a1 = a2b2 * a0a1a2
								Tensor<Tdata> D_tmp1 = Tensor_Multiply::x1y0y1_ax1_y0y1a(D_a2b2, D_a);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp1), D_mul);
							}
							if(D_mul.empty())	continue;

							// D_result = D_mul * D_a1b1 * D_b
							for(const TAC &Ab01 : list_Ab01)
							{
								if(this->filter_atom->filter_for32(label,Aa01,Ab2,Ab01))	continue;
								const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
								if(D_b.empty())	continue;
								const Tensor<Tdata> &D_a1b1 = tools.get_Ds_ab(Label::ab::a1b1, Aa01, Ab01);
								if(D_a1b1.empty())	continue;

								// b1b2a0 = a1b1 * b2a0a1
								const Tensor<Tdata> D_tmp2 = Tensor_Multiply::x1y0y1_ax1_y0y1a(D_a1b1, D_mul);
								// a0b0 = b1b2a0 * b0b1b2
								Tensor<Tdata> D_tmp3 = Tensor_Multiply::x2y0_abx2_y0ab(D_tmp2, D_b);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp3), Ds_result_fixed[Ab01]);
							}
						} // end for Ab01

						if(!Ds_result_fixed.empty())
							LRI_Cal_Aux::add_Ds(std::move(Ds_result_fixed),
												Ds_result_thread[Aa01]);
						LRI_Cal_Aux::add_Ds_omp_try_map(Ds_result_thread, Ds_result, lock_Ds_result_add_map, fac_add_Ds);
					} // end for Aa01
				} break; // end case a1b1_a2b2

			  // Aab_Aab::a01b2_a2b01

				case Label::ab_ab::a1b2_a2b1:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a1b2).Ds_ab );
					const std::vector<TAC> &list_Aa2 = LRI_Cal_Aux::filter_list_map(
						list_Aa2_Da,
						data_wrapper(Label::ab::a2b1).Ds_ab );
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a2b1).index_Ds_ab[0]);
					const std::vector<TAC>  list_Ab2 = LRI_Cal_Aux::filter_list_set(
						list_Ab2_Db,
						data_wrapper(Label::ab::a1b2).index_Ds_ab[0]);

					for(const TA &Aa01 : list_Aa01)
					{
						if(this->filter_atom->filter_for1(label,Aa01))	continue;
						#pragma omp for schedule(dynamic) nowait
						for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
						{
							const TAC &Ab01 = list_Ab01[ib01];
							if(this->filter_atom->filter_for2(label,Aa01,Ab01))	continue;
							// D_mul1 = D_b * D_a1b2
							Tensor<Tdata> D_mul1;
							for(const TAC &Ab2 : list_Ab2)
							{
								if(this->filter_atom->filter_for31(label,Aa01,Ab01,Ab2))	continue;
								const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
								if(D_b.empty())	continue;
								const Tensor<Tdata> &D_a1b2 = tools.get_Ds_ab(Label::ab::a1b2, Aa01, Ab2);
								if(D_a1b2.empty())	continue;

								// b0b1a1 = b0b1b2 * a1b2
								Tensor<Tdata> D_tmp1 = Tensor_Multiply::x0x1y0_x0x1a_y0a(D_b, D_a1b2);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp1), D_mul1);
							}
							if(D_mul1.empty())	continue;

							// D_mul2 = D_a2b1 * D_a
							Tensor<Tdata> D_mul2;
							for(const TAC &Aa2 : list_Aa2)
							{
								if(this->filter_atom->filter_for32(label,Aa01,Ab01,Aa2))	continue;
								const Tensor<Tdata> &D_a = Global_Func::find(Ds_a_transpose, Aa01, Aa2);
								if(D_a.empty())	continue;
								const Tensor<Tdata> &D_a2b1 = tools.get_Ds_ab(Label::ab::a2b1, Aa2, Ab01);
								if(D_a2b1.empty())	continue;
								// b1a1a0 = a2b1 * a1a0a2
								Tensor<Tdata> D_tmp2 = Tensor_Multiply::x1y0y1_ax1_y0y1a(D_a2b1, D_a);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp2), D_mul2);
							}
							if(D_mul2.empty())	continue;

							// D_result = D_mul2 * D_mul1
							// a0b0 = b1a1a0 * b0b1a1
							Tensor<Tdata> D_mul3 = Tensor_Multiply::x2y0_abx2_y0ab(D_mul2, D_mul1);
							LRI_Cal_Aux::add_Ds(std::move(D_mul3),
												Ds_result_thread[Aa01][Ab01]);
						} // end for Aa01

						LRI_Cal_Aux::add_Ds_omp_try_map(Ds_result_thread, Ds_result, lock_Ds_result_add_map, fac_add_Ds);
					} // end for Ab01
				} break; // end case a1b2_a2b1

				case Label::ab_ab::a0b2_a2b0:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a0b2).Ds_ab );
					const std::vector<TAC> &list_Aa2 = LRI_Cal_Aux::filter_list_map(
						list_Aa2_Da,
						data_wrapper(Label::ab::a2b0).Ds_ab );
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a2b0).index_Ds_ab[0]);
					const std::vector<TAC>  list_Ab2 = LRI_Cal_Aux::filter_list_set(
						list_Ab2_Db,
						data_wrapper(Label::ab::a0b2).index_Ds_ab[0]);

					for(const TA &Aa01 : list_Aa01)
					{
						if(this->filter_atom->filter_for1(label,Aa01))	continue;
						#pragma omp for schedule(dynamic) nowait
						for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
						{
							const TAC &Ab01 = list_Ab01[ib01];
							if(this->filter_atom->filter_for2(label,Aa01,Ab01))	continue;
							// D_mul1 = D_b * D_a0b2
							Tensor<Tdata> D_mul1;
							for(const TAC &Ab2 : list_Ab2)
							{
								if(this->filter_atom->filter_for31(label,Aa01,Ab01,Ab2))	continue;
								const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
								if(D_b.empty())	continue;
								const Tensor<Tdata> &D_a0b2 = tools.get_Ds_ab(Label::ab::a0b2, Aa01, Ab2);
								if(D_a0b2.empty())	continue;

								// a0b0b1 = a0b2 * b0b1b2
								Tensor<Tdata> D_tmp1 = Tensor_Multiply::x0y0y1_x0a_y0y1a(D_a0b2, D_b);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp1), D_mul1);
							}
							if(D_mul1.empty())	continue;

							// D_mul2 = D_a2b0 * D_a
							Tensor<Tdata> D_mul2;
							for(const TAC &Aa2 : list_Aa2)
							{
								if(this->filter_atom->filter_for32(label,Aa01,Ab01,Aa2))	continue;
								const Tensor<Tdata> &D_a = Global_Func::find(Ds_a_transpose, Aa01, Aa2);
								if(D_a.empty())	continue;
								const Tensor<Tdata> &D_a2b0 = tools.get_Ds_ab(Label::ab::a2b0, Aa2, Ab01);
								if(D_a2b0.empty())	continue;
								// a1a0b0 = a1a0a2 * a2b0
								Tensor<Tdata> D_tmp2 = Tensor_Multiply::x0x1y1_x0x1a_ay1(D_a, D_a2b0);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp2), D_mul2);
							}
							if(D_mul2.empty())	continue;

							// D_result = D_mul2 * D_mul1
							// b1a1 = a1a0b0 * a0b0b1
							Tensor<Tdata> D_mul3 = Tensor_Multiply::x0y2_x0ab_aby2(D_mul2, D_mul1);
							LRI_Cal_Aux::add_Ds(std::move(D_mul3),
												Ds_result_thread[Aa01][Ab01]);
						} // end for Aa01

						LRI_Cal_Aux::add_Ds_omp_try_map(Ds_result_thread, Ds_result, lock_Ds_result_add_map, fac_add_Ds);
					} // end for Ab01
				} break; // end case a0b2_a2b0

				case Label::ab_ab::a0b2_a2b1:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a0b2).Ds_ab );
					const std::vector<TAC> &list_Aa2 = LRI_Cal_Aux::filter_list_map(
						list_Aa2_Da,
						data_wrapper(Label::ab::a2b1).Ds_ab );
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a2b1).index_Ds_ab[0]);
					const std::vector<TAC>  list_Ab2 = LRI_Cal_Aux::filter_list_set(
						list_Ab2_Db,
						data_wrapper(Label::ab::a0b2).index_Ds_ab[0]);

					for(const TA &Aa01 : list_Aa01)
					{
						if(this->filter_atom->filter_for1(label,Aa01))	continue;
						#pragma omp for schedule(dynamic) nowait
						for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
						{
							const TAC &Ab01 = list_Ab01[ib01];
							if(this->filter_atom->filter_for2(label,Aa01,Ab01))	continue;
							// D_mul1 = D_b * D_a0b2
							Tensor<Tdata> D_mul1;
							for(const TAC &Ab2 : list_Ab2)
							{
								if(this->filter_atom->filter_for31(label,Aa01,Ab01,Ab2))	continue;
								const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
								if(D_b.empty())	continue;
								const Tensor<Tdata> &D_a0b2 = tools.get_Ds_ab(Label::ab::a0b2, Aa01, Ab2);
								if(D_a0b2.empty())	continue;

								// b0b1a0 = b0b1b2 * a0b2
								Tensor<Tdata> D_tmp1 = Tensor_Multiply::x0x1y0_x0x1a_y0a(D_b, D_a0b2);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp1), D_mul1);
							}
							if(D_mul1.empty())	continue;

							// D_mul2 = D_a2b1 * D_a
							Tensor<Tdata> D_mul2;
							for(const TAC &Aa2 : list_Aa2)
							{
								if(this->filter_atom->filter_for32(label,Aa01,Ab01,Aa2))	continue;
								const Tensor<Tdata> &D_a = tools.get_Ds_ab(Label::ab::a, Aa01, Aa2);
								if(D_a.empty())	continue;
								const Tensor<Tdata> &D_a2b1 = tools.get_Ds_ab(Label::ab::a2b1, Aa2, Ab01);
								if(D_a2b1.empty())	continue;
								// b1a0a1 = a2b1 * a0a1a2
								Tensor<Tdata> D_tmp2 = Tensor_Multiply::x1y0y1_ax1_y0y1a(D_a2b1, D_a);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp2), D_mul2);
							}
							if(D_mul2.empty())	continue;

							// D_result = D_mul2 * D_mul1
							// a1b0 = b1a0a1 * b0b1a0
							Tensor<Tdata> D_mul3 = Tensor_Multiply::x2y0_abx2_y0ab(D_mul2, D_mul1);
							LRI_Cal_Aux::add_Ds(std::move(D_mul3),
												Ds_result_thread[Aa01][Ab01]);
						} // end for Aa01

						LRI_Cal_Aux::add_Ds_omp_try_map(Ds_result_thread, Ds_result, lock_Ds_result_add_map, fac_add_Ds);
					} // end for Ab01
				} break; // end case a0b2_a2b1

				case Label::ab_ab::a1b2_a2b0:
				{
					const std::vector<TA >  list_Aa01 = LRI_Cal_Aux::filter_list_map(
						list_Aa01_Da,
						data_wrapper(Label::ab::a1b2).Ds_ab );
					const std::vector<TAC> &list_Aa2 = LRI_Cal_Aux::filter_list_map(
						list_Aa2_Da,
						data_wrapper(Label::ab::a2b0).Ds_ab );
					const std::vector<TAC>  list_Ab01 = LRI_Cal_Aux::filter_list_set(
						list_Ab01_Db,
						data_wrapper(Label::ab::a2b0).index_Ds_ab[0]);
					const std::vector<TAC>  list_Ab2 = LRI_Cal_Aux::filter_list_set(
						list_Ab2_Db,
						data_wrapper(Label::ab::a1b2).index_Ds_ab[0]);

					for(const TA &Aa01 : list_Aa01)
					{
						if(this->filter_atom->filter_for1(label,Aa01))	continue;
						#pragma omp for schedule(dynamic) nowait
						for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
						{
							const TAC &Ab01 = list_Ab01[ib01];
							if(this->filter_atom->filter_for2(label,Aa01,Ab01))	continue;
							// D_mul1 = D_b * D_a1b2
							Tensor<Tdata> D_mul1;
							for(const TAC &Ab2 : list_Ab2)
							{
								if(this->filter_atom->filter_for31(label,Aa01,Ab01,Ab2))	continue;
								const Tensor<Tdata> &D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
								if(D_b.empty())	continue;
								const Tensor<Tdata> &D_a1b2 = tools.get_Ds_ab(Label::ab::a1b2, Aa01, Ab2);
								if(D_a1b2.empty())	continue;

								// a1b0b1 = a1b2 * b0b1b2
								Tensor<Tdata> D_tmp1 = Tensor_Multiply::x0y0y1_x0a_y0y1a(D_a1b2, D_b);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp1), D_mul1);
							}
							if(D_mul1.empty())	continue;

							// D_mul2 = D_a2b0 * D_a
							Tensor<Tdata> D_mul2;
							for(const TAC &Aa2 : list_Aa2)
							{
								if(this->filter_atom->filter_for32(label,Aa01,Ab01,Aa2))	continue;
								const Tensor<Tdata> &D_a = tools.get_Ds_ab(Label::ab::a, Aa01, Aa2);
								if(D_a.empty())	continue;
								const Tensor<Tdata> &D_a2b0 = tools.get_Ds_ab(Label::ab::a2b0, Aa2, Ab01);
								if(D_a2b0.empty())	continue;
								// a0a1b0 = a0a1a2 * a2b0
								Tensor<Tdata> D_tmp2 = Tensor_Multiply::x0x1y1_x0x1a_ay1(D_a, D_a2b0);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp2), D_mul2);
							}
							if(D_mul2.empty())	continue;

							// D_result = D_mul2 * D_mul1
							// a0b1 = a0a1b0 * a1b0b1
							Tensor<Tdata> D_mul3 = Tensor_Multiply::x0y2_x0ab_aby2(D_mul2, D_mul1);
							LRI_Cal_Aux::add_Ds(std::move(D_mul3),
												Ds_result_thread[Aa01][Ab01]);
						} // end for Aa01

						LRI_Cal_Aux::add_Ds_omp_try_map(Ds_result_thread, Ds_result, lock_Ds_result_add_map, fac_add_Ds);
					} // end for Ab01
				} break; // end case a1b2_a2b0

				default:
					throw std::invalid_argument(std::string(__FILE__)+std::to_string(__LINE__));
			} // end switch(label)
		} // end for label

		LRI_Cal_Aux::add_Ds_omp_wait_map(Ds_result_thread, Ds_result, lock_Ds_result_add_map, fac_add_Ds);
		#pragma omp critical(ri_weighted_short_stats)
		{
			weighted_short_candidates += weighted_short_candidates_thread;
			weighted_short_skips += weighted_short_skips_thread;
			weighted_short_max_score = std::max(weighted_short_max_score, weighted_short_max_score_thread);
		}
	} // end #pragma omp parallel

	LRI_Cal_Aux::destroy_lock_result(lock_Ds_result_add_map, Ds_result);
	lri_set_weighted_short_screen_stats(
		weighted_short_lri_key,
		{weighted_short_candidates, weighted_short_skips, weighted_short_max_score});

  #ifdef __MKL_RI
	mkl_set_num_threads(mkl_threads);
  #endif
}	// end LRI::cal_loop3()

}	// end namespace RI

