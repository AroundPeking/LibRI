// ===================
//  Author: Peize Lin
//  date: 2022.08.12
// ===================

#pragma once

#include "Label.h"
#include "RI/global/Tensor.h"
#include "RI/global/Global_Func-1.h"
#include "RI/global/Array_Operator.h"

#include <map>
#include <unordered_map>

namespace RI
{

	template<typename TA, typename TC, typename Tdata>
	struct Data_Pack;

template<typename TA, typename TC, typename Tdata>
class LRI_Cal_Tools
{
public:
	using TAC = std::pair<TA,TC>;

	LRI_Cal_Tools(
		const TC &period_in,
		const std::map<std::string, Data_Pack<TA,TC,Tdata>> &data_pool,
		const std::unordered_map<Label::ab, std::string> &data_ab_name,		
		std::vector<std::map<TA, std::map<TAC, Tensor<Tdata>>>> &Ds_result_in)
			:period(period_in), Ds_result(Ds_result_in)
	{
		this->Ds_ab_ptr.reserve(Label::array_ab.size());
		for(const auto &name : data_ab_name)
		{
			if(!name.second.empty())
				this->Ds_ab_ptr[name.first] = &data_pool.at(name.second).Ds_ab;
		}
	}

	inline const Tensor<Tdata> &get_Ds_ab(
		const Label::ab &label,
		const TA &Aa, const TAC &Ab) const
	{
		return Global_Func::find(
			*this->Ds_ab_ptr.at(label),
			Aa, Ab);
	}
	inline const Tensor<Tdata> &get_Ds_ab(
		const Label::ab &label,
		const TAC &Aa, const TAC &Ab) const
	{
		using namespace Array_Operator;
		return Global_Func::find(
			*this->Ds_ab_ptr.at(label),
			Aa.first, TAC{Ab.first, (Ab.second-Aa.second)%this->period});
	}

	inline Tensor<Tdata> &get_D_result(const std::size_t icoef, const TA &Aa, const TAC &Ab)
	{
		return this->Ds_result[icoef][Aa][Ab];
	}

	inline Tensor<Tdata> &get_D_result(const std::size_t icoef, const TAC &Aa, const TAC &Ab)
	{
		using namespace Array_Operator;
		return this->Ds_result[icoef][Aa.first][TAC{Ab.first, (Ab.second-Aa.second)%period}];
	}

	inline static std::vector<Label::ab> split_b01(const Label::ab_ab &label)
	{
		switch(label)
		{
			case Label::ab_ab::a0b0_a2b2:	case Label::ab_ab::a0b0_a1b2:	return {Label::ab::a0b0};
			case Label::ab_ab::a0b1_a1b2:	case Label::ab_ab::a0b1_a2b2:	return {Label::ab::a0b1};
			case Label::ab_ab::a1b0_a2b2:	case Label::ab_ab::a0b2_a1b0:	return {Label::ab::a1b0};
			case Label::ab_ab::a1b1_a2b2:	case Label::ab_ab::a0b2_a1b1:	return {Label::ab::a1b1};
			case Label::ab_ab::a0b2_a2b0:	case Label::ab_ab::a1b2_a2b0:	return {Label::ab::a2b0};
			case Label::ab_ab::a0b2_a2b1:	case Label::ab_ab::a1b2_a2b1:	return {Label::ab::a2b1};
			case Label::ab_ab::a1b0_a2b1:									return {Label::ab::a1b0, Label::ab::a2b1};
			case Label::ab_ab::a1b1_a2b0:									return {Label::ab::a1b1, Label::ab::a2b0};
			case Label::ab_ab::a0b0_a2b1:									return {Label::ab::a0b0, Label::ab::a2b1};
			case Label::ab_ab::a0b1_a2b0:									return {Label::ab::a0b1, Label::ab::a2b0};
			case Label::ab_ab::a0b0_a1b1:									return {Label::ab::a0b0, Label::ab::a1b1};
			case Label::ab_ab::a0b1_a1b0:									return {Label::ab::a0b1, Label::ab::a1b0};
			default:	throw std::invalid_argument(std::string(__FILE__)+" line "+std::to_string(__LINE__));
		}
	}

	std::vector<Label::ab_ab> filter_Ds_Ab01(
		const std::vector<Label::ab_ab> &labels,
		const TA &Aa01, const TAC &Aa2, const TAC &Ab01) const
	{
		const auto judge_label = [this, &Aa01, &Aa2, &Ab01](const Label::ab_ab &label_ab_ab) -> bool
		{
			const std::vector<Label::ab> &labels_ab = split_b01(label_ab_ab);
			for(const Label::ab &label_ab : labels_ab)
			{
				const int a = Label::get_a(label_ab);
				switch(a)
				{
					case 0:	case 1:
						if(!this->get_Ds_ab(label_ab, Aa01, Ab01).empty())
							return true;
						else
							continue;
					case 2:
						if(!this->get_Ds_ab(label_ab, Aa2, Ab01).empty())
							return true;
						else
							continue;
					default:
						throw std::invalid_argument(std::string(__FILE__)+" line "+std::to_string(__LINE__));
				}
			}
			return false;
		};

		std::vector<Label::ab_ab> labels_filter;
		for(const Label::ab_ab &label : labels)
		{
			if(judge_label(label))
				labels_filter.push_back(label);
		}
		return labels_filter;
	}	

public:		// private:
	const TC &period;
	std::unordered_map<Label::ab, const std::map<TA, std::map<TAC, Tensor<Tdata>>>*> Ds_ab_ptr;
	std::vector<std::map<TA, std::map<TAC, Tensor<Tdata>>>> &Ds_result;
};

}