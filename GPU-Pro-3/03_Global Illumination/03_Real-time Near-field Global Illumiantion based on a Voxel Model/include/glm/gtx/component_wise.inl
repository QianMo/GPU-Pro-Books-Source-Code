///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2007-05-21
// Updated : 2007-05-21
// Licence : This source is under MIT License
// File    : gtx_component_wise.inl
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace glm{
namespace gtx{
namespace component_wise
{
	template <typename genType>
	inline typename genType::value_type compAdd(const genType& v)
	{
        typename genType::size_type result = typename genType::value_type(0);
		for(typename genType::size_type i = 0; i < genType::value_size(); ++i)
			result += v[i];
		return result;
	}
/*
	template <typename genType>
	inline valType compAdd(const genType& v)
	{
		valType result = valType(0);
		for(sizeType i = 0; i < valSize; ++i)
			result += v[i];
		return result;
	}
*/
	template <typename genType>
	inline typename genType::value_type compMul(const genType& v)
	{
        typename genType::value_type result = typename genType::value_type(1);
		for(typename genType::size_type i = 0; i < genType::value_size(); ++i)
			result *= v[i];
		return result;
	}
/*
	template <typename genType>
	inline typename genType::value_type compMul(const genType& v)
	{
		valType result = valType(0);
		for(GLMsizeType i = 0; i < valSize; ++i)
			result *= v[i];
		return result;
	}
*/
	template <typename genType>
	inline typename genType::value_type compMin(const genType& v)
	{
        typename genType::value_type result = typename genType::value_type(v[0]);
		for(typename genType::size_type i = 1; i < genType::value_size(); ++i)
			result = min(result, v[i]);
		return result;
	}
/*
	template <typename genType>
	inline typename genType::value_type compMin(const genType& v)
	{
		valType result = valType(0);
		for(GLMsizeType i = 0; i < valSize; ++i)
			result = min(result, v[i]);
		return result;
	}
*/
	template <typename genType>
	inline typename genType::value_type compMax(const genType& v)
	{
        typename genType::value_type result = typename genType::value_type(v[0]);
		for(typename genType::size_type i = 1; i < genType::value_size(); ++i)
			result = max(result, v[i]);
		return result;
	}
/*
	template <typename genType>
	inline typename genType::value_type compMax(const genType& v)
	{
		GLMvalType result = GLMvalType(0);
		for(GLMsizeType i = 0; i < GLMvalSize; ++i)
			result = max(result, v[i]);
		return result;
	}
*/
}//namespace component_wise
}//namespace gtx
}//namespace glm
