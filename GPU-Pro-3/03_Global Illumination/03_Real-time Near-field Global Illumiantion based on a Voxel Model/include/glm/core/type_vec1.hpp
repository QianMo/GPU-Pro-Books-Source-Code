///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2008-08-25
// Updated : 2008-08-31
// Licence : This source is under MIT License
// File    : glm/core/type_vec1.h
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_core_type_gentype1
#define glm_core_type_gentype1

#include "type_float.hpp"
#include "type_int.hpp"
#include "type_size.hpp"
#include "_swizzle.hpp"

namespace glm
{
	namespace test
	{
		void main_vec1();
	}//namespace test

	namespace detail
	{
		template <typename T> struct tref1;
		template <typename T> struct tref2;
		template <typename T> struct tref3;
		template <typename T> struct tref4;
		template <typename T> struct tvec1;
		template <typename T> struct tvec2;
		template <typename T> struct tvec3;
		template <typename T> struct tvec4;

		template <typename valType>
		struct tvec1
		{
			//////////////////////////////////////
			// Typedef (Implementation details)

			typedef valType value_type;
			typedef valType& value_reference;
			typedef valType* value_pointer;
			typedef tvec1<bool> bool_type;

			typedef glm::sizeType size_type;
			static size_type value_size();
			static bool is_vector();

			typedef tvec1<valType> type;
			typedef tvec1<valType>* pointer;
			typedef const tvec1<valType>* const_pointer;
			typedef const tvec1<valType>*const const_pointer_const;
			typedef tvec1<valType>*const pointer_const;
			typedef tvec1<valType>& reference;
			typedef const tvec1<valType>& const_reference;
			typedef const tvec1<valType>& param_type;

			//////////////////////////////////////
			// Data

#	if defined(GLM_USE_ONLY_XYZW)
			value_type x, y;
#	else//GLM_USE_ONLY_XYZW
			union {value_type x, r, s;};
#	endif//GLM_USE_ONLY_XYZW

			//////////////////////////////////////
			// Accesses

			valType& operator[](size_type i);
			valType const & operator[](size_type i) const;

			//////////////////////////////////////
			// Address (Implementation details)

			value_type const * _address() const{return (value_type*)(this);}
			value_type * _address(){return (value_type*)(this);}

			//////////////////////////////////////
			// Implicit basic constructors

			tvec1();
			tvec1(const tvec1<valType>& v);

			//////////////////////////////////////
			// Explicit basic constructors

			tvec1(valType s);

			//////////////////////////////////////
			// Swizzle constructors

			tvec1(const tref1<valType>& r);

			//////////////////////////////////////
			// Convertion scalar constructors

			//! Explicit converions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename U> 
			explicit tvec1(U x);

			//////////////////////////////////////
			// Convertion vector constructors

			//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename U> 
			explicit tvec1(const tvec2<U>& v);
			//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename U> 
			explicit tvec1(const tvec3<U>& v);
			//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename U> 
			explicit tvec1(const tvec4<U>& v);

			//////////////////////////////////////
			// Unary arithmetic operators

			tvec1<valType>& operator= (const tvec1<valType>& v);

			tvec1<valType>& operator+=(valType const & s);
			tvec1<valType>& operator+=(const tvec1<valType>& v);
			tvec1<valType>& operator-=(valType const & s);
			tvec1<valType>& operator-=(const tvec1<valType>& v);
			tvec1<valType>& operator*=(valType const & s);
			tvec1<valType>& operator*=(const tvec1<valType>& v);
			tvec1<valType>& operator/=(valType const & s);
			tvec1<valType>& operator/=(const tvec1<valType>& v);
			tvec1<valType>& operator++();
			tvec1<valType>& operator--();

			//////////////////////////////////////
			// Unary bit operators

			tvec1<valType>& operator%=(valType const & s);
			tvec1<valType>& operator%=(const tvec1<valType>& v);
			tvec1<valType>& operator&=(valType const & s);
			tvec1<valType>& operator&=(const tvec1<valType>& v);
			tvec1<valType>& operator|=(valType const & s);
			tvec1<valType>& operator|=(const tvec1<valType>& v);
			tvec1<valType>& operator^=(valType const & s);
			tvec1<valType>& operator^=(const tvec1<valType>& v);
			tvec1<valType>& operator<<=(valType const & s);
			tvec1<valType>& operator<<=(const tvec1<valType>& v);
			tvec1<valType>& operator>>=(valType const & s);
			tvec1<valType>& operator>>=(const tvec1<valType>& v);

			//////////////////////////////////////
			// Swizzle operators

			valType swizzle(comp X) const;
			tvec2<valType> swizzle(comp X, comp Y) const;
			tvec3<valType> swizzle(comp X, comp Y, comp Z) const;
			tvec4<valType> swizzle(comp X, comp Y, comp Z, comp W) const;
			tref1<valType> swizzle(comp X);
		};

		template <typename T>
		struct tref1
		{
			tref1(T& x);
			tref1(const tref1<T>& r);
			tref1(const tvec1<T>& v);

			tref1<T>& operator= (const tref1<T>& r);
			tref1<T>& operator= (const tvec1<T>& v);

			T& x;
		};
	} //namespace detail

	namespace core{
	namespace type{
	namespace vector{

	//////////////////////////
	// Boolean definition

	typedef detail::tvec1<bool>				bvec1;

	//////////////////////////
	// Float definition

#ifndef GLM_PRECISION 
	typedef detail::tvec1<mediump_float>	vec1;
#elif(GLM_PRECISION & GLM_PRECISION_HIGHP_FLOAT)
	typedef detail::tvec1<highp_float>		vec1;
#elif(GLM_PRECISION & GLM_PRECISION_MEDIUMP_FLOAT)
	typedef detail::tvec1<mediump_float>	vec1;
#elif(GLM_PRECISION & GLM_PRECISION_LOWP_FLOAT)
	typedef detail::tvec1<lowp_float>		vec1;
#else
	typedef detail::tvec1<mediump_float>	vec1;
#endif//GLM_PRECISION

	namespace precision
	{
		typedef detail::tvec1<highp_float>		highp_vec1;
		typedef detail::tvec1<mediump_float>	mediump_vec1;
		typedef detail::tvec1<lowp_float>		lowp_vec1;
	}
	//namespace precision

	//////////////////////////
	// Signed integer definition

#ifndef GLM_PRECISION 
	typedef detail::tvec1<mediump_int>		ivec1;
#elif(GLM_PRECISION & GLM_PRECISION_HIGHP_INT)
	typedef detail::tvec1<highp_int>		ivec1;
#elif(GLM_PRECISION & GLM_PRECISION_MEDIUMP_INT)
	typedef detail::tvec1<mediump_int>		ivec1;
#elif(GLM_PRECISION & GLM_PRECISION_LOWP_INT)
	typedef detail::tvec1<lowp_int>			ivec1;
#endif//GLM_PRECISION

	namespace precision
	{
		typedef detail::tvec1<highp_int>		highp_ivec1;
		typedef detail::tvec1<mediump_int>		mediump_ivec1;
		typedef detail::tvec1<lowp_int>			lowp_ivec1;
	}
	//namespace precision

	//////////////////////////
	// Unsigned integer definition

#ifndef GLM_PRECISION 
	typedef detail::tvec1<mediump_uint>		uvec1;
#elif(GLM_PRECISION & GLM_PRECISION_HIGHP_UINT)
	typedef detail::tvec1<highp_uint>		uvec1;
#elif(GLM_PRECISION & GLM_PRECISION_MEDIUMP_UINT)
	typedef detail::tvec1<mediump_uint>		uvec1;
#elif(GLM_PRECISION & GLM_PRECISION_LOWP_UINT)
	typedef detail::tvec1<lowp_uint>		uvec1;
#endif//GLM_PRECISION

	namespace precision
	{
		typedef detail::tvec1<highp_uint>		highp_uvec1;
		typedef detail::tvec1<mediump_uint>		mediump_uvec1;
		typedef detail::tvec1<lowp_uint>		lowp_uvec1;
	}
	//namespace precision

	}//namespace vector
	}//namespace type
	}//namespace core
}//namespace glm

#include "type_vec1.inl"

#endif//glm_core_type_gentype1
