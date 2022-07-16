// ===================
//  Author: Peize Lin
//  date: 2021.10.31
// ===================

#pragma once

#include "Blas_Interface-Contiguous.h"
#include "Tensor.h"
#include <cassert>
#include <string>

namespace Blas_Interface
{
	// nrm2 = ||x||_2
	template<typename T>
	static inline T nrm2(const Tensor<std::complex<T>> &X)
	{
		return nrm2(X.get_shape_all(), X.ptr());
	}
	template<typename T>
	static inline T nrm2(const Tensor<T> &X)
	{
		return nrm2(X.get_shape_all(), X.ptr());
	}

	// d = Vx * Vy
	template<typename T>
	static inline T dot(const Tensor<T> &X, const Tensor<T> &Y)
	{
		assert(X.shape.size()==1);
		assert(Y.shape.size()==1);
		assert(X.shape[0] == Y.shape[0]);

		return dot(X.shape[0], X.ptr(), Y.ptr());
	}

	// Vy = alpha * Ma.? * Vx + beta * Vy
	template<typename T>
	static inline void gemv(const char transA, 
		const double alpha, const Tensor<T> &A, const Tensor<T> &X,
		const double beta, Tensor<T> &Y)
	{
		assert(A.shape.size()==2);
		assert(X.shape.size()==1);
		assert(Y.shape.size()==1);
		if(transA=='N')
		{
			assert(A.shape[0]==Y.shape[0]);
			assert(A.shape[1]==X.shape[0]);
		}
		else if(transA=='T' || transA=='C')
		{
			assert(A.shape[1]==Y.shape[0]);
			assert(A.shape[0]==X.shape[0]);
		}
		else
		{
			throw std::invalid_argument("trans must be 'N', 'T' or 'C'.\nIn "+std::string(__FILE__)+" line "+std::to_string(__LINE__)+". ");
		}

		gemv(transA, A.shape[0], A.shape[1],
			alpha, A.ptr(), X.ptr(),
			beta, Y.ptr());
	}
	// Vy = alpha * Ma.? * Vx
	template<typename T>
	static inline Tensor<T> gemv(const char transA, 
		const double alpha, const Tensor<T> &A, const Tensor<T> &X)
	{
		constexpr double beta = 0.0;
		Tensor<double> Y({(transA=='N') ? A.shape[0] : A.shape[1]});
		gemv(transA, alpha, A, X, beta, Y);
		return Y;
	}

	// Mc = alpha * Ma.? * Mb.? + beta * Mc
	template<typename T>
	static inline void gemm(const char transA, const char transB,
		const double alpha, const Tensor<T> &A, const Tensor<T> &B,
		const double beta, Tensor<T> &C)
	{
		assert(A.shape.size()==2);
		assert(B.shape.size()==2);
		assert(C.shape.size()==2);
		if(transA=='N')
			assert(A.shape[0]==C.shape[0]);
		else if(transA=='T' || transA=='C')
			assert(A.shape[1]==C.shape[0]);
		else
			throw std::invalid_argument("trans must be 'N', 'T' or 'C'.\nIn "+std::string(__FILE__)+" line "+std::to_string(__LINE__)+". ");
		if(transB=='N')
			assert(B.shape[1]==C.shape[1]);
		else if(transB=='T' || transB=='C')
			assert(B.shape[0]==C.shape[1]);
		else
			throw std::invalid_argument("trans must be 'N', 'T' or 'C'.\nIn "+std::string(__FILE__)+" line "+std::to_string(__LINE__)+". ");

		const size_t m = C.shape[0];
		const size_t n = C.shape[1];
		const size_t k = (transA=='N') ? A.shape[1] : A.shape[0];

		if(transB=='N')
			assert(k==B.shape[0]);
		else
			assert(k==B.shape[1]);
			
		gemm(transA, transB, m, n, k,
			alpha, A.ptr(), B.ptr(),
			beta, C.ptr());
	}
	// Mc = alpha * Ma.? * Mb.?
	template<typename T>
	static inline Tensor<T> gemm(const char transA, const char transB,
		const double alpha, const Tensor<T> &A, const Tensor<T> &B)
	{
		constexpr double beta = 0.0;
		Tensor<T> C({
			(transA=='N') ? A.shape[0] : A.shape[1],
			(transB=='N') ? B.shape[1] : B.shape[0] });
		gemm(transA, transB,
			alpha, A, B,
			beta, C);
		return C;
	}

	// Mc = alpha * Ma   * Ma.T + beta * C
	// Mc = alpha * Ma.T * Ma   + beta * C
	template<typename T>
	static inline void syrk(const char uploC, const char transA,
		const double alpha, const Tensor<T> &A,
		const double beta, Tensor<T> &C)
	{
		assert(A.shape.size()==2);
		assert(C.shape.size()==2);
		assert(C.shape[0]==C.shape[1]);
		if(transA=='N')
			assert(A.shape[0]==C.shape[0]);
		else if(transA=='T' || transA=='C')
			assert(A.shape[1]==C.shape[0]);
		else
			throw std::invalid_argument("trans must be 'N', 'T' or 'C'.\nIn "+std::string(__FILE__)+" line "+std::to_string(__LINE__)+". ");

		const size_t n = C.shape[0];
		const size_t k = (transA=='N') ? A.shape[1] : A.shape[0];
		syrk(uploC, transA, n, k,
			alpha, A.ptr(),
			beta, C.ptr());
	}
	// Mc = alpha * Ma   * Ma.T
	// Mc = alpha * Ma.T * Ma  
	template<typename T>
	static inline Tensor<T> syrk(const char uploC, const char transA,
		const double alpha, const Tensor<T> &A)
	{
		constexpr double beta = 0.0;
		const size_t n = (transA=='N') ? A.shape[0] : A.shape[1];
		Tensor<T> C({n,n});
		syrk(uploC, transA,	alpha, A, beta, C);
		return C;
	}
}

#include "Tensor.hpp"
#include "Tensor-multiply.hpp"