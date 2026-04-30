// ===================
//  Author: Peize Lin
//  date: 2023.08.03
// ===================

#pragma once

#include "Tensor.h"
#include "Blas_Interface-Contiguous.h"

#include <stdexcept>
#include <string>

namespace RI
{

namespace Tensor_Multiply
{

namespace detail
{
	inline std::string shape_to_string(const Shape_Vector &shape)
	{
		std::string result = "[";
		for(std::size_t i=0; i<shape.size(); ++i)
		{
			if(i)	result += "x";
			result += std::to_string(shape[i]);
		}
		result += "]";
		return result;
	}

	inline void require_contract_dims(const bool ok,
		const char *func_name,
		const Shape_Vector &tx_shape,
		const Shape_Vector &ty_shape,
		const char *rule)
	{
		if(ok)	return;
		throw std::runtime_error(
			std::string("Tensor_Multiply::") + func_name
			+ " dimension mismatch (" + rule + "), Tx="
			+ shape_to_string(tx_shape) + ", Ty=" + shape_to_string(ty_shape));
	}
}

}

}

#include "Tensor_Multiply-22.hpp"
#include "Tensor_Multiply-23.hpp"
#include "Tensor_Multiply-32.hpp"
#include "Tensor_Multiply-33.hpp"
