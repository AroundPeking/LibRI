// ===================
//  Author: Peize Lin
//  date: 2023.08.03
// ===================

#pragma once

#include "Tensor_Multiply.h"

namespace RI
{

namespace Tensor_Multiply
{
	// Txy(x1,x2,y1) = Tx(a,x1,x2) * Ty(a,y1)
	template<typename Tdata>
	Tensor<Tdata> x1x2y1_x0y0(const Tensor<Tdata> &Tx, const Tensor<Tdata> &Ty)
	{
		assert(Tx.shape.size()==3);
		assert(Ty.shape.size()==2);
		assert(Tx.shape[0]==Ty.shape[0]);
		const std::size_t x12 = Tx.shape[1] * Tx.shape[2];
		Tensor<Tdata> Txy({Tx.shape[1], Tx.shape[2], Ty.shape[1]});
		Blas_Interface::gemm(
			'T', 'N', x12, Ty.shape[1], Tx.shape[0],
			Tdata(1.0), Tx.ptr(), Ty.ptr(),
			Tdata(0.0), Txy.ptr());
		return Txy;
	}

	// Txy(x0,x1,y0) = Tx(x0,x1,a) * Ty(y0,a)
	template<typename Tdata>
	Tensor<Tdata> x0x1y0_x2y1(const Tensor<Tdata> &Tx, const Tensor<Tdata> &Ty)
	{
		assert(Tx.shape.size()==3);
		assert(Ty.shape.size()==2);
		assert(Tx.shape[2]==Ty.shape[1]);
		const std::size_t x01 = Tx.shape[0] * Tx.shape[1];
		Tensor<Tdata> Txy({Tx.shape[0], Tx.shape[1], Ty.shape[0]});
		Blas_Interface::gemm(
			'N', 'T', x01, Ty.shape[0], Tx.shape[2],
			Tdata(1.0), Tx.ptr(), Ty.ptr(),
			Tdata(0.0), Txy.ptr());
		return Txy;
	}

	// Txy(x1,x2,y0) = Tx(a,x1,x2) * Ty(y0,a)
	template<typename Tdata>
	Tensor<Tdata> x1x2y0_x0y1(const Tensor<Tdata> &Tx, const Tensor<Tdata> &Ty)
	{
		assert(Tx.shape.size()==3);
		assert(Ty.shape.size()==2);
		const std::size_t x12 = Tx.shape[1] * Tx.shape[2];
		Tensor<Tdata> Txy({Tx.shape[1], Tx.shape[2], Ty.shape[0]});
		Blas_Interface::gemm(
			'T', 'T', x12, Ty.shape[0], Tx.shape[0],
			Tdata(1.0), Tx.ptr(), Ty.ptr(),
			Tdata(0.0), Txy.ptr());
		return Txy;
	}

	// Txy(x0,x1,y1) = Tx(x0,x1,a) * Ty(a,y1)
	template<typename Tdata>
	Tensor<Tdata> x0x1y1_x2y0(const Tensor<Tdata> &Tx, const Tensor<Tdata> &Ty)
	{
		assert(Tx.shape.size()==3);
		assert(Ty.shape.size()==2);
		const std::size_t x01 = Tx.shape[0] * Tx.shape[1];
		Tensor<Tdata> Txy({Tx.shape[0], Tx.shape[1], Ty.shape[1]});
		Blas_Interface::gemm(
			'N', 'N', x01, Ty.shape[1], Tx.shape[2],
			Tdata(1.0), Tx.ptr(), Ty.ptr(),
			Tdata(0.0), Txy.ptr());
		return Txy;
	}
}

}