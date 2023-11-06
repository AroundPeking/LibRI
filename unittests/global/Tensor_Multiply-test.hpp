// ===================
//  Author: Peize Lin
//  date: 2023.08.03
// ===================

#pragma once

#include "RI/global/Tensor_Multiply.h"

namespace Tensor_Multiply_Test
{
	template<typename Tdata>
	RI::Tensor<Tdata> init_tensor(const RI::Shape_Vector &shape)
	{
		RI::Tensor<Tdata> T(shape);
		const std::size_t size = T.get_shape_all();
		for(std::size_t i=0; i<size; ++i)
			T.ptr()[i] = i;
		return T;
	}
}

#include "Tensor_Multiply-23-test.hpp"
#include "Tensor_Multiply-32-test.hpp"
#include "Tensor_Multiply-33-test.hpp"				

namespace Tensor_Multiply_Test
{
	template<typename Tdata>
	void main()
	{
		x1y0y1_ax1_y0y1a_test<Tdata>();
		x1x2y1_ax1x2_ay1_test<Tdata>();
		x0x1y0_x0x1a_y0a_test<Tdata>();
		x1x2y0_ax1x2_y0a_test<Tdata>();
		x0x1y1_x0x1a_ay1_test<Tdata>();
		x0y2_x0ab_aby2_test<Tdata>();
		x2y0_abx2_y0ab_test<Tdata>();
	}
}
