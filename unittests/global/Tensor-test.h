// ===================
//  Author: Peize Lin
//  date: 2021.10.31
// ===================

#pragma once

#include "global/Tensor.h"

template<typename T>
std::ostream &operator<<(std::ostream &os, const Tensor<T> &t)
{
	if(t.shape.size()==1)
	{
		for(size_t i0=0; i0<t.shape[0]; ++i0)
			os<<t(i0)<<"\t";
		os<<std::endl;
	}
	else if(t.shape.size()==2)
	{
		for(size_t i0=0; i0<t.shape[0]; ++i0)
		{
			for(size_t i1=0; i1<t.shape[1]; ++i1)
				os<<t(i0,i1)<<"\t";
			os<<std::endl;
		}
	}
	else if(t.shape.size()==3)
	{
		os<<"["<<std::endl;
		for(size_t i0=0; i0<t.shape[0]; ++i0)
		{
			for(size_t i1=0; i1<t.shape[1]; ++i1)
			{
				for(size_t i2=0; i2<t.shape[2]; ++i2)
					os<<t(i0,i1,i2)<<"\t";
				os<<std::endl;
			}
			os<<std::endl;
		}
		os<<"]"<<std::endl;
	}
	return os;
}
