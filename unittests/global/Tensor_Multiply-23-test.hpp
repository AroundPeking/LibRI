// ===================
//  Author: Peize Lin
//  date: 2023.08.03
// ===================

#pragma once

#include "Tensor_Multiply-test.hpp"

#define FOR_0(i,N)					for(std::size_t i=0; i<N; ++i)

namespace Tensor_Multiply_Test
{
	// Txy(x1,y0,y1) = Tx(a,x1) * Ty(y0,y1,a)
	template<typename Tdata>
	void x1y0y1_ax1_y0y1a_test()
	{
		const std::size_t X1=2, Y0=3, Y1=4, A=5;
		const RI::Tensor<Tdata> Tx = init_tensor<Tdata>({A,X1});
		const RI::Tensor<Tdata> Ty = init_tensor<Tdata>({Y0,Y1,A});
		const RI::Tensor<Tdata> Txy = RI::Tensor_Multiply::x1y0y1_ax1_y0y1a(Tx,Ty);
		FOR_0(x1,X1)
			FOR_0(y0,Y0)
				FOR_0(y1,Y1)
				{
					Tdata result = 0;
					FOR_0(a,A)
						result += Tx(a,x1) * Ty(y0,y1,a);
					assert(Txy(x1,y0,y1) == result);
				}
	}

}

#undef FOR_0
