// ===================
//  Author: Peize Lin
//  date: 2023.08.02
// ===================

#pragma once

#include "LRI.h"
#include "LRI_Cal_Aux.h"
#include "../global/Array_Operator.h"
#include "../global/Tensor_Multiply.h"

#include <omp.h>
#ifdef __MKL_RI
#include <mkl_service.h>
#endif

namespace RI
{

template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
void LRI<TA,Tcell,Ndim,Tdata>::cal_loop3(
	const std::vector<Label::ab_ab> &labels,
	std::vector<std::map<TA, std::map<TAC, Tensor<Tdata>>>> &Ds_result,
	const double fac_add_Ds)
{
	using namespace Array_Operator;

	if(Ds_result.empty())
		Ds_result.resize(1);

	const bool flag_D_b_transpose = [&labels]() -> bool
	{
		for(const Label::ab_ab &label : labels)
			switch(label)
			{
				case Label::ab_ab::a0b0_a2b1:	case Label::ab_ab::a0b0_a2b2:
					return true;
			}
		return false;
	}();
	const bool flag_D_a_transpose = [&labels]() -> bool
	{
		for(const Label::ab_ab &label : labels)
			switch(label)
			{
				case Label::ab_ab::a0b0_a1b2:	case Label::ab_ab::a0b0_a2b2:
					return true;
			}
		return false;
	}();
	const std::map<TA, std::map<TAC, Tensor<Tdata>>> Ds_a_transpose
		= flag_D_a_transpose
		? LRI_Cal_Aux::cal_Ds_transpose(this->Ds_ab[Label::ab::a])
		: std::map<TA, std::map<TAC, Tensor<Tdata>>>{};
	const std::map<TA, std::map<TAC, Tensor<Tdata>>> Ds_b_transpose
		= flag_D_b_transpose
		? LRI_Cal_Aux::cal_Ds_transpose(this->Ds_ab[Label::ab::b])
		: std::map<TA, std::map<TAC, Tensor<Tdata>>>{};

	omp_lock_t lock_Ds_result_add;
	omp_init_lock(&lock_Ds_result_add);

#ifdef __MKL_RI
	const std::size_t mkl_threads = mkl_get_max_threads();
	mkl_set_num_threads(1);
#endif

	#pragma omp parallel
	{
		std::vector<std::map<TA, std::map<TAC, Tensor<Tdata>>>> Ds_result_thread(1);
		LRI_Cal_Tools<TA,TC,Tdata> tools(this->period, this->Ds_ab, Ds_result_thread);

		for(const Label::ab_ab &label : labels)
		{
			switch(label)
			{
				case Label::ab_ab::a0b0_a1b1:
				{
					for(const TAC &Aa2 : this->parallel->get_list_Aa2())
					{
						std::map<TAC,Tensor<Tdata>> Ds_result_fixed;

						const std::vector<TAC> &list_Ab01 = this->parallel->get_list_Ab01();
						#pragma omp for schedule(dynamic) nowait
						for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
						{
							const TAC &Ab01 = list_Ab01[ib01];
							// D_mul = D_a * D_a0b0 * D_a1b1
							Tensor<Tdata> D_mul;
							for(const TA &Aa01 : this->parallel->get_list_Aa01())
							{
								const Tensor<Tdata> D_a = tools.get_Ds_ab(Label::ab::a, Aa01, Aa2);
								if(D_a.empty())	continue;
								const Tensor<Tdata> D_a0b0 = tools.get_Ds_ab(Label::ab::a0b0, Aa01, Ab01);
								if(D_a0b0.empty())	continue;
								const Tensor<Tdata> D_a1b1 = tools.get_Ds_ab(Label::ab::a1b1, Aa01, Ab01);
								if(D_a1b1.empty())	continue;

								// a1a2b0 = a0a1a2 * a0b0
								const Tensor<Tdata> D_tmp1 = Tensor_Multiply::x1x2y1_x0y0(D_a, D_a0b0);
								// a2b0b1 = a1a2b0 * a1b1
								Tensor<Tdata> D_tmp2 = Tensor_Multiply::x1x2y1_x0y0(D_tmp1, D_a1b1);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp2), D_mul);
							}
							if(D_mul.empty())	continue;

							// D_result = D_mul * D_b
							for(const TAC &Ab2 : this->parallel->get_list_Ab2())
							{
								const Tensor<Tdata> D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
								if(D_b.empty())	continue;
								// a2b2 = a2b0b1 * b0b1b2
								Tensor<Tdata> D_tmp3 = Tensor_Multiply::x0y2_x1y0_x2y1(D_mul, D_b);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp3), Ds_result_fixed[Ab2]);
							}
						} // end for Ab01

						if(!Ds_result_fixed.empty())
							LRI_Cal_Aux::add_Ds( LRI_Cal_Aux::Ds_translate(std::move(Ds_result_fixed), Aa2.second, this->period),
							                     Ds_result_thread[0][Aa2.first]);
						LRI_Cal_Aux::add_Ds_omp_try(std::move(Ds_result_thread), Ds_result, lock_Ds_result_add, fac_add_Ds);
					} // end for Ab2
					break;
				} // end case a0b0_a1b1

				case Label::ab_ab::a0b0_a1b2:
				{
					for(const TAC &Ab01 : this->parallel->get_list_Ab01())
					{
						std::map<TAC,Tensor<Tdata>> Ds_result_fixed;

						const std::vector<TA> &list_Aa01 = this->parallel->get_list_Aa01();
						#pragma omp for schedule(dynamic) nowait
						for(std::size_t ia01=0; ia01<list_Aa01.size(); ++ia01)
						{
							const TA &Aa01 = list_Aa01[ia01];
							const Tensor<Tdata> D_a0b0 = tools.get_Ds_ab(Label::ab::a0b0, Aa01, Ab01);
							if(D_a0b0.empty())	continue;
							// D_mul = D_b * D_a1b2
							Tensor<Tdata> D_mul;
							for(const TAC &Ab2 : this->parallel->get_list_Ab2())
							{
								const Tensor<Tdata> D_b = tools.get_Ds_ab(Label::ab::b, Ab01, Ab2);
								if(D_b.empty())	continue;
								const Tensor<Tdata> D_a1b2 = tools.get_Ds_ab(Label::ab::a1b2, Aa01, Ab2);
								if(D_a1b2.empty())	continue;

								// b0b1a1 = b0b1b2 * a1b2
								Tensor<Tdata> D_tmp1 = Tensor_Multiply::x0x1y0_x2y1(D_b, D_a1b2);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp1), D_mul);
							}
							if(D_mul.empty())	continue;

							// D_result = D_mul * D_a * D_a0b0
							for(const TAC &Aa2 : this->parallel->get_list_Aa2())
							{
								const Tensor<Tdata> &D_a_transpose = Global_Func::find(Ds_a_transpose, Aa01, Aa2);
								if(D_a_transpose.empty())	continue;
								// b1a1a0 = b0b1a1 * a0b0
								const Tensor<Tdata> D_tmp2 = Tensor_Multiply::x1x2y0_x0y1(D_mul, D_a0b0);
								// a2b1 = a1a0a2 * b1a1a0
								Tensor<Tdata> D_tmp3 = Tensor_Multiply::x2y0_x0y1_x1y2(D_a_transpose, D_tmp2);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp3), Ds_result_fixed[Aa2]);
							}
						} // end for Aa01

						if(!Ds_result_fixed.empty())
							LRI_Cal_Aux::add_Ds( LRI_Cal_Aux::Ds_exchange(std::move(Ds_result_fixed), Ab01, this->period),
							                     Ds_result_thread[0]);
						LRI_Cal_Aux::add_Ds_omp_try(std::move(Ds_result_thread), Ds_result, lock_Ds_result_add, fac_add_Ds);
					} // end for Ab01
					break;
				} // end case a0b0_a1b2				

				case Label::ab_ab::a0b0_a2b1:
				{
					for(const TA &Aa01 : this->parallel->get_list_Aa01())
					{
						std::map<TAC,Tensor<Tdata>> Ds_result_fixed;

						const std::vector<TAC> &list_Ab01 = this->parallel->get_list_Ab01();
						#pragma omp for schedule(dynamic) nowait
						for(std::size_t ib01=0; ib01<list_Ab01.size(); ++ib01)
						{
							const TAC &Ab01 = list_Ab01[ib01];
							const Tensor<Tdata> D_a0b0 = tools.get_Ds_ab(Label::ab::a0b0, Aa01, Ab01);
							if(D_a0b0.empty())	continue;
							// D_mul = D_a * D_a2b1
							Tensor<Tdata> D_mul;
							for(const TAC &Aa2 : this->parallel->get_list_Aa2())
							{
								const Tensor<Tdata> D_a = tools.get_Ds_ab(Label::ab::a, Aa01, Aa2);
								if(D_a.empty())	continue;
								const Tensor<Tdata> D_a2b1 = tools.get_Ds_ab(Label::ab::a2b1, Aa2, Ab01);
								if(D_a2b1.empty())	continue;

								// a0a1b1 = a0a1a2 * a2b1
								Tensor<Tdata> D_tmp1 = Tensor_Multiply::x0x1y1_x2y0(D_a, D_a2b1);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp1), D_mul);
							}
							if(D_mul.empty())	continue;

							// D_result = D_mul * D_a0b0 * D_b
							for(const TAC &Ab2 : this->parallel->get_list_Ab2())
							{
								const Tensor<Tdata> &D_b_transpose = Global_Func::find(Ds_b_transpose, Ab01.first, TAC{Ab2.first, (Ab2.second-Ab01.second)%this->period});
								if(D_b_transpose.empty())	continue;

								// a1b1b0 = a0a1b1 * a0b0
								const Tensor<Tdata> D_tmp2 = Tensor_Multiply::x1x2y1_x0y0(D_mul, D_a0b0);
								// a1b2 = a1b1b0 * b1b0b2
								Tensor<Tdata> D_tmp3 = Tensor_Multiply::x0y2_x1y0_x2y1(D_tmp2, D_b_transpose);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp3), Ds_result_fixed[Ab2]);
							}
						} // end for Ab01

						if(!Ds_result_fixed.empty())
							LRI_Cal_Aux::add_Ds( std::move(Ds_result_fixed),
							                     Ds_result_thread[0][Aa01]);
						LRI_Cal_Aux::add_Ds_omp_try(std::move(Ds_result_thread), Ds_result, lock_Ds_result_add, fac_add_Ds);
					} // end for Aa01
					break;
				} // end case a0b0_a2b1

				case Label::ab_ab::a0b0_a2b2:
				{
					for(const TA &Aa01 : this->parallel->get_list_Aa01())
					{
						std::map<TAC,Tensor<Tdata>> Ds_result_fixed;

						const std::vector<TAC> &list_Ab2 = this->parallel->get_list_Ab2();
						#pragma omp for schedule(dynamic) nowait
						for(std::size_t ib2=0; ib2<list_Ab2.size(); ++ib2)
						{
							const TAC &Ab2 = list_Ab2[ib2];
							// D_mul = D_a * D_a2b2
							Tensor<Tdata> D_mul;
							for(const TAC &Aa2 : this->parallel->get_list_Aa2())
							{
								const Tensor<Tdata> &D_a_transpose = Global_Func::find(Ds_a_transpose, Aa01, Aa2);
								if(D_a_transpose.empty())	continue;
								const Tensor<Tdata> D_a2b2 = tools.get_Ds_ab(Label::ab::a2b2, Aa2, Ab2);
								if(D_a2b2.empty())	continue;

								// b2a1a0 = a2b2 * a1a0a2
								Tensor<Tdata> D_tmp1 = Tensor_Multiply::x1y0y1_x0y2(D_a2b2, D_a_transpose);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp1), D_mul);
							}
							if(D_mul.empty())	continue;

							// D_result = D_mul * D_a0b0 * D_b
							for(const TAC &Ab01 : this->parallel->get_list_Ab01())
							{
								const Tensor<Tdata> &D_b_transpose = Global_Func::find(Ds_b_transpose, Ab01.first, TAC{Ab2.first, (Ab2.second-Ab01.second)%this->period});
								if(D_b_transpose.empty())	continue;
								const Tensor<Tdata> D_a0b0 = tools.get_Ds_ab(Label::ab::a0b0, Aa01, Ab01);
								if(D_a0b0.empty())	continue;

								// b0b2a1 = a0b0 * b2a1a0
								const Tensor<Tdata> D_tmp2 = Tensor_Multiply::x1y0y1_x0y2(D_a0b0, D_mul);
								// a1b1 = b0b2a1 * b1b0b2
								Tensor<Tdata> D_tmp3 = Tensor_Multiply::x2y0_x0y1_x1y2(D_tmp2, D_b_transpose);
								LRI_Cal_Aux::add_Ds(std::move(D_tmp3), Ds_result_fixed[Ab01]);
							}
						} // end for Ab01

						if(!Ds_result_fixed.empty())
							LRI_Cal_Aux::add_Ds( std::move(Ds_result_fixed),
							                     Ds_result_thread[0][Aa01]);
						LRI_Cal_Aux::add_Ds_omp_try(std::move(Ds_result_thread), Ds_result, lock_Ds_result_add, fac_add_Ds);
					} // end for Aa01
					break;
				} // end case a0b0_a2b2

				default:
					throw std::invalid_argument(std::string(__FILE__)+std::to_string(__LINE__));
			} // end switch(label)
		} // end for label

		LRI_Cal_Aux::add_Ds_omp_wait(std::move(Ds_result_thread), Ds_result, lock_Ds_result_add, fac_add_Ds);
	} // end #pragma omp parallel

	omp_destroy_lock(&lock_Ds_result_add);
#ifdef __MKL_RI
	mkl_set_num_threads(mkl_threads);
#endif
}

}