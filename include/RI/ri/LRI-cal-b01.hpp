// ===================
//  Author: Peize Lin
//  date: 2022.08.12
// ===================

#pragma once

#include "LRI.h"
#include "LRI_Cal_Aux.h"

template<typename TA, typename Tcell, size_t Ndim, typename Tdata>
void LRI<TA,Tcell,Ndim,Tdata>::set_cal_funcs_b01()
{
	#define Macro_cal_func_b01(D_ab_first_in, D_ab_second_in, D_mul2_in, D_mul3_in, D_mul4_in, D_result_in)	\
	[this](	\
		const Label::ab_ab &label,	\
		const TA &Aa01, const TAC &Aa2, const TAC &Ab01, const TAC &Ab2,	\
		const Tensor<Tdata> &D_a, const Tensor<Tdata> &D_b, const Tensor<Tdata> &D_b_transpose,	\
		std::unordered_map<Label::ab_ab, Tensor<Tdata>> &Ds_b01,	\
		std::unordered_map<Label::ab_ab, Tdata_real> &Ds_b01_csm,	\
		LRI_Cal_Tools<TA,TC,Tdata> &tools)	\
	{	\
		const Tensor<Tdata> &D_ab_first  = D_ab_first_in;	\
		const Tensor<Tdata> &D_ab_second = D_ab_second_in;	\
		if(!D_ab_first)		return;	\
		if(!D_ab_second)	return;	\
		typename CS_Matrix<TA,TC,Tdata_real>::Step csm_step;	\
		if(this->csm.threshold.at(label))	\
			csm_step = this->csm.set_label_A(label, Aa01, Aa2, Ab01, Ab2, period);	\
		\
		if(this->csm.threshold.at(label))	\
			if(this->csm.filter_4(csm_step))	\
				return;	\
		if(Ds_b01[label].empty())	\
		{	\
			const Tensor<Tdata> D_mul2 = D_mul2_in;	\
			\
			if(this->csm.threshold.at(label)){	\
				const Tdata_real D_mul2_csm = D_mul2.norm(2);	\
				if(this->csm.filter_3(csm_step, D_mul2_csm))	\
					return;	\
			}	\
			Ds_b01[label] = D_mul3_in;	\
			\
			if(this->csm.threshold.at(label))	\
				Ds_b01_csm[label] = Ds_b01[label].norm(2);	\
		}	\
		if(this->csm.threshold.at(label))	\
			if(this->csm.filter_2(csm_step, Ds_b01_csm[label]))	\
				return;	\
		const Tensor<Tdata> &D_mul3 = Ds_b01[label];	\
		const Tensor<Tdata> D_mul4 = D_mul4_in;	\
		\
		if(this->csm.threshold.at(label))	\
		{	\
			const Tdata_real D_mul4_csm = D_mul4.norm(2);	\
			if(this->csm.filter_1(csm_step, D_mul4_csm))	\
				return;	\
		}	\
		Tensor<Tdata> &D_result = D_result_in;	\
		LRI_Cal_Aux::add_D(D_mul4, D_result);	\
	};	\


	this->cal_funcs[Label::ab_ab::a0b0_a1b1] = Macro_cal_func_b01(
		tools.get_Ds_ab(Label::ab::a0b0, Aa01, Ab01),
		tools.get_Ds_ab(Label::ab::a1b1, Aa01, Ab01),
		// D_mul2(ia1,ia2,ib0) = D_a(ia0,ia1,ia2) * D_ab(ia0,ib0)
		Blas_Interface::gemm(
			'T', 'N', Tdata(1),
			LRI_Cal_Aux::tensor3_merge(D_a,false),
			D_ab_first
			).reshape({D_a.shape[1], D_a.shape[2], D_b.shape[0]}),
		// D_mul3(ia2,ib0,ib1) = D_mul2(ia1,ia2,ib0) * D_ab(ia1,ib1)
		Blas_Interface::gemm(
			'T', 'N', Tdata(1),
			LRI_Cal_Aux::tensor3_merge(D_mul2,false),
			D_ab_second
			).reshape({D_a.shape[2], D_b.shape[0], D_b.shape[1]}),
		// D_mul4(ia2,ib2) = D_mul3(ia2,ib0,ib1) * D_b(ib0,ib1,ib2)
		Blas_Interface::gemm(
			'N', 'N', Tdata(1),
			LRI_Cal_Aux::tensor3_merge(D_mul3,false),
			LRI_Cal_Aux::tensor3_merge(D_b,true)),
		tools.get_D_result(Aa2, Ab2));

	this->cal_funcs[Label::ab_ab::a0b1_a1b0] = Macro_cal_func_b01(
		tools.get_Ds_ab(Label::ab::a1b0, Aa01, Ab01),
		tools.get_Ds_ab(Label::ab::a0b1, Aa01, Ab01),
		// D_mul2(ia0,ia2,ib0) = D_a(ia0,ia1,ia2) * D_ab(ia1,ib0)
		Blas_Interface::gemm(
			'T', 'N', Tdata(1),
			LRI_Cal_Aux::tensor3_merge(LRI_Cal_Aux::tensor3_transpose(D_a),false),
			D_ab_first
			).reshape({D_a.shape[0], D_a.shape[2], D_b.shape[0]}),
		// D_mul3(ia2,ib0,ib1) = D_mul2(ia0,ia2,ib0) * D_ab(ia0,ib1)
		Blas_Interface::gemm(
			'T', 'N', Tdata(1),
			LRI_Cal_Aux::tensor3_merge(D_mul2,false),
			D_ab_second
			).reshape({D_a.shape[2], D_b.shape[0], D_b.shape[1]}),
		// D_mul4(ia2,ib2) = D_mul3(ia2,ib0,ib1) * D_b(ib0,ib1,ib2)
		Blas_Interface::gemm(
			'N', 'N', Tdata(1),
			LRI_Cal_Aux::tensor3_merge(D_mul3,false),
			LRI_Cal_Aux::tensor3_merge(D_b,true)),
		tools.get_D_result(Aa2, Ab2));

	this->cal_funcs[Label::ab_ab::a0b0_a2b1] = Macro_cal_func_b01(
		tools.get_Ds_ab(Label::ab::a0b0, Aa01, Ab01),
		tools.get_Ds_ab(Label::ab::a2b1, Aa2, Ab01),
		// D_mul2(ib0,ia1,ia2) = D_ab(ia0,ib0) * D_a(ia0,ia1,ia2)
		Blas_Interface::gemm(
			'T', 'N', Tdata(1),
			D_ab_first,
			LRI_Cal_Aux::tensor3_merge(D_a,false)
			).reshape({D_b.shape[0], D_a.shape[1], D_a.shape[2]}),
		// D_mul3(ia1,ib0,ib1) = D_mul2(ib0,ia1,ia2) * D_ab(ia2,ib1)
		Blas_Interface::gemm(
			'N', 'N', Tdata(1),
			LRI_Cal_Aux::tensor3_merge(LRI_Cal_Aux::tensor3_transpose(D_mul2),true),
			D_ab_second
			).reshape({D_a.shape[1], D_b.shape[0], D_b.shape[1]}),
		// D_mul4(ia1,ib2) = D_mul3(ia1,ib0,ib1) * D_b(ib0,ib1,ib2)
		Blas_Interface::gemm(
			'N', 'N', Tdata(1),
			LRI_Cal_Aux::tensor3_merge(D_mul3,false),
			LRI_Cal_Aux::tensor3_merge(D_b,true)),
		tools.get_D_result(Aa01, Ab2));

	this->cal_funcs[Label::ab_ab::a0b1_a2b0] = Macro_cal_func_b01(
		tools.get_Ds_ab(Label::ab::a2b0, Aa2, Ab01),
		tools.get_Ds_ab(Label::ab::a0b1, Aa01, Ab01),
		// D_mul2(ia0,ia1,ib0) = D_a(ia0,ia1,ia2) * D_ab(ia2,ib0)
		Blas_Interface::gemm(
			'N', 'N', Tdata(1),
			LRI_Cal_Aux::tensor3_merge(D_a,true),
			D_ab_first
			).reshape({D_a.shape[0], D_a.shape[1], D_b.shape[0]}),
		// D_mul3(ia1,ib0,ib1) = D_mul2(ia0,ia1,ib0) * D_ab(ia0,ib1)
		Blas_Interface::gemm(
			'T', 'N', Tdata(1),
			LRI_Cal_Aux::tensor3_merge(D_mul2,false),
			D_ab_second
			).reshape({D_a.shape[1], D_b.shape[0], D_b.shape[1]}),
		// D_mul4(ia1,ib2) = D_mul3(ia1,ib0,ib1) * D_b(ib0,ib1,ib2)
		Blas_Interface::gemm(
			'N', 'N', Tdata(1),
			LRI_Cal_Aux::tensor3_merge(D_mul3,false),
			LRI_Cal_Aux::tensor3_merge(D_b,true)),
		tools.get_D_result(Aa01, Ab2));

	this->cal_funcs[Label::ab_ab::a1b0_a2b1] = Macro_cal_func_b01(
		tools.get_Ds_ab(Label::ab::a2b1, Aa2, Ab01),
		tools.get_Ds_ab(Label::ab::a1b0, Aa01, Ab01),
		// D_mul2(ib1,ia0,ia1) = D_ab(ia2,ib1) * D_a(ia0,ia1,ia2)
		Blas_Interface::gemm(
			'T', 'T', Tdata(1),
			D_ab_first,
			LRI_Cal_Aux::tensor3_merge(D_a,true)
			).reshape({D_b.shape[1], D_a.shape[0], D_a.shape[1]}),
		// D_mul3(ib0,ib1,ia0) = D_ab(ia1,ib0) * D_mul2(ib1,ia0,ia1)
		Blas_Interface::gemm(
			'T', 'T', Tdata(1),
			D_ab_second,
			LRI_Cal_Aux::tensor3_merge(D_mul2,true)
			).reshape({D_b.shape[0], D_b.shape[1], D_a.shape[0]}),
		// D_mul4(ia0,ib2) = D_mul3(ib0,ib1,ia0) * D_b(ib0,ib1,ib2)
		Blas_Interface::gemm(
			'T', 'N', Tdata(1),
			LRI_Cal_Aux::tensor3_merge(D_mul3,true),
			LRI_Cal_Aux::tensor3_merge(D_b,true)),
		tools.get_D_result(Aa01, Ab2));

	this->cal_funcs[Label::ab_ab::a1b1_a2b0] = Macro_cal_func_b01(
		tools.get_Ds_ab(Label::ab::a2b0, Aa2, Ab01),
		tools.get_Ds_ab(Label::ab::a1b1, Aa01, Ab01),
		// D_mul2(ib0,ia0,ia1) = D_ab(ia2,ib0) * D_a(ia0,ia1,ia2)
		Blas_Interface::gemm(
			'T', 'T', Tdata(1),
			D_ab_first,
			LRI_Cal_Aux::tensor3_merge(D_a,true)
			).reshape({D_b.shape[0], D_a.shape[0], D_a.shape[1]}),
		// D_mul3(ia0,ib0,ib1) = D_mul2(ib0,ia0,ia1) * D_ab(ia1,ib1)
		Blas_Interface::gemm(
			'N', 'N', Tdata(1),
			LRI_Cal_Aux::tensor3_merge(LRI_Cal_Aux::tensor3_transpose(D_mul2),true),
			D_ab_second
			).reshape({D_a.shape[0], D_b.shape[0], D_b.shape[1]}),
		// D_mul4(ia0,ib2) = D_mul3(ia0,ib0,ib1) * D_b(ib0,ib1,ib2)
		Blas_Interface::gemm(
			'N', 'N', Tdata(1),
			LRI_Cal_Aux::tensor3_merge(D_mul3,false),
			LRI_Cal_Aux::tensor3_merge(D_b,true)),
		tools.get_D_result(Aa01, Ab2));


	#undef Macro_cal_func_b01
}