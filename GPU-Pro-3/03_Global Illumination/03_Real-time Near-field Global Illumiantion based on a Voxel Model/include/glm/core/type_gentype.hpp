///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2008-10-05
// Updated : 2008-10-05
// Licence : This source is under MIT License
// File    : glm/core/type_gentype.h
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_core_type_gentype
#define glm_core_type_gentype

#include "type_size.hpp"

namespace glm{

enum profile
{
	nice,
	fast,
	simd
};

namespace detail{

template <typename valTypeT, uint colT, uint rowT, profile proT = nice>
class genType
{
public:
	//////////////////////////////////////
	// Traits

	typedef sizeType							size_type;
	typedef valTypeT							value_type;

	typedef genType<value_type, colT, rowT>		class_type;

	typedef genType<bool, colT, rowT>			bool_type;
	typedef genType<value_type, rowT, 1>		col_type;
	typedef genType<value_type, colT, 1>		row_type;
	typedef genType<value_type, rowT, colT>		transpose_type;

	static size_type							col_size();
	static size_type							row_size();
	static size_type							value_size();
	static bool									is_scalar();
	static bool									is_vector();
	static bool									is_matrix();

private:
	// Data 
	col_type value[colT];		

public:
	//////////////////////////////////////
	// Constructors
	genType();
	genType(class_type const & m);

	explicit genType(value_type const & x);
	explicit genType(value_type const * const x);
	explicit genType(col_type const * const x);

	//////////////////////////////////////
	// Conversions
	template <typename vU, uint cU, uint rU, profile pU>
	explicit genType(genType<vU, cU, rU, pU> const & m);

	//////////////////////////////////////
	// Accesses
	col_type& operator[](size_type i);
	col_type const & operator[](size_type i) const;

	//////////////////////////////////////
	// Unary updatable operators
	class_type& operator=  (class_type const & x);
	class_type& operator+= (value_type const & x);
	class_type& operator+= (class_type const & x);
	class_type& operator-= (value_type const & x);
	class_type& operator-= (class_type const & x);
	class_type& operator*= (value_type const & x);
	class_type& operator*= (class_type const & x);
	class_type& operator/= (value_type const & x);
	class_type& operator/= (class_type const & x);
	class_type& operator++ ();
	class_type& operator-- ();
};

}//namespace detail
}//namespace glm

#include "type_gentype.inl"

#endif//glm_core_type_gentype
