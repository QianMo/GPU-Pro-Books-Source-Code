///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2008-08-18
// Updated : 2008-08-31
// Licence : This source is under MIT License
// File    : glm/core/type_tvec2.h
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_core_type_gentype2
#define glm_core_type_gentype2

#include "type_float.hpp"
#include "type_int.hpp"
#include "type_size.hpp"
#include "_swizzle.hpp"

namespace glm
{
	namespace test
	{
		void main_vec2();
	}
	//namespace test

	namespace detail
	{
		template <typename T> struct tref2;
		template <typename T> struct tref3;
		template <typename T> struct tref4;
		template <typename T> struct tvec3;
		template <typename T> struct tvec4;

		template <typename valType>
		struct tvec2
		{
			//////////////////////////////////////
			// Typedef (Implementation details)

			typedef valType value_type;
			typedef valType& value_reference;
			typedef valType* value_pointer;
			typedef tvec2<bool> bool_type;

			typedef glm::sizeType size_type;
			static size_type value_size();
			static bool is_vector();

			typedef tvec2<valType> type;
			typedef tvec2<valType>* pointer;
			typedef const tvec2<valType>* const_pointer;
			typedef const tvec2<valType>*const const_pointer_const;
			typedef tvec2<valType>*const pointer_const;
			typedef tvec2<valType>& reference;
			typedef const tvec2<valType>& const_reference;
			typedef const tvec2<valType>& param_type;

			//////////////////////////////////////
			// Data

#	if defined(GLM_USE_ONLY_XYZW)
			value_type x, y;
#	else//GLM_USE_ONLY_XYZW
#		ifdef GLM_USE_ANONYMOUS_UNION
			union 
			{
				struct{value_type x, y;};
				struct{value_type r, g;};
				struct{value_type s, t;};
			};
#		else//GLM_USE_ANONYMOUS_UNION
			union {value_type x, r, s;};
			union {value_type y, g, t;};
#		endif//GLM_USE_ANONYMOUS_UNION
#	endif//GLM_USE_ONLY_XYZW

			//////////////////////////////////////
			// Accesses

			valType & operator[](size_type i);
			valType const & operator[](size_type i) const;

			//////////////////////////////////////
			// Address (Implementation details)

			value_type const * _address() const{return (value_type*)(this);}
			value_type * _address(){return (value_type*)(this);}

			//////////////////////////////////////
			// Implicit basic constructors

			tvec2();
			tvec2(tvec2<valType> const & v);

			//////////////////////////////////////
			// Explicit basic constructors

			explicit tvec2(valType s);
			explicit tvec2(valType s1, valType s2);

			//////////////////////////////////////
			// Swizzle constructors

			tvec2(tref2<valType> const & r);

			//////////////////////////////////////
			// Convertion scalar constructors

			//! Explicit converions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename U> 
			explicit tvec2(U x);
			//! Explicit converions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename U, typename V> 
			explicit tvec2(U x, V y);

			//////////////////////////////////////
			// Convertion vector constructors

			//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename U> 
			explicit tvec2(tvec2<U> const & v);
			//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename U> 
			explicit tvec2(tvec3<U> const & v);
			//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename U> 
			explicit tvec2(tvec4<U> const & v);

			//////////////////////////////////////
			// Unary arithmetic operators

			tvec2<valType>& operator= (tvec2<valType> const & v);

			tvec2<valType>& operator+=(valType const & s);
			tvec2<valType>& operator+=(tvec2<valType> const & v);
			tvec2<valType>& operator-=(valType const & s);
			tvec2<valType>& operator-=(tvec2<valType> const & v);
			tvec2<valType>& operator*=(valType const & s);
			tvec2<valType>& operator*=(tvec2<valType> const & v);
			tvec2<valType>& operator/=(valType const & s);
			tvec2<valType>& operator/=(tvec2<valType> const & v);
			tvec2<valType>& operator++();
			tvec2<valType>& operator--();

			//////////////////////////////////////
			// Unary bit operators

			tvec2<valType>& operator%=(valType const & s);
			tvec2<valType>& operator%=(tvec2<valType> const & v);
			tvec2<valType>& operator&=(valType const & s);
			tvec2<valType>& operator&=(tvec2<valType> const & v);
			tvec2<valType>& operator|=(valType const & s);
			tvec2<valType>& operator|=(tvec2<valType> const & v);
			tvec2<valType>& operator^=(valType const & s);
			tvec2<valType>& operator^=(tvec2<valType> const & v);
			tvec2<valType>& operator<<=(valType const & s);
			tvec2<valType>& operator<<=(tvec2<valType> const & v);
			tvec2<valType>& operator>>=(valType const & s);
			tvec2<valType>& operator>>=(tvec2<valType> const & v);

			//////////////////////////////////////
			// Swizzle operators

			valType swizzle(comp X) const;
			tvec2<valType> swizzle(comp X, comp Y) const;
			tvec3<valType> swizzle(comp X, comp Y, comp Z) const;
			tvec4<valType> swizzle(comp X, comp Y, comp Z, comp W) const;
			tref2<valType> swizzle(comp X, comp Y);
		};

//		tvec2<glm::detail::thalf>::tvec2<glm::detail::thalf><float,float>(float x, float y) :
//			x(detail::thalf(x)),
//			y(detail::thalf(y))
//		{}

		template <typename T>
		struct tref2
		{
			tref2(T& x, T& y);
			tref2(tref2<T> const & r);
			tref2(tvec2<T> const & v);

			tref2<T>& operator= (tref2<T> const & r);
			tref2<T>& operator= (tvec2<T> const & v);

			T& x;
			T& y;
		};
	} //namespace detail

	namespace core{
	namespace type{
	namespace vector{

	//////////////////////////
	// Boolean definition

	//! 2 components vector of boolean. 
	//! From GLSL 1.30.8 specification, section 4.1.5 Vectors.
	typedef detail::tvec2<bool>				bvec2;

	//////////////////////////
	// Float definition

#ifndef GLM_PRECISION 
	//! 2 components vector of floating-point numbers. 
	//! From GLSL 1.30.8 specification, section 4.1.5 Vectors.
	typedef detail::tvec2<mediump_float>	vec2;
#elif(GLM_PRECISION & GLM_PRECISION_HIGHP_FLOAT)
	typedef detail::tvec2<highp_float>		vec2;
#elif(GLM_PRECISION & GLM_PRECISION_MEDIUMP_FLOAT)
	typedef detail::tvec2<mediump_float>	vec2;
#elif(GLM_PRECISION & GLM_PRECISION_LOWP_FLOAT)
	typedef detail::tvec2<lowp_float>		vec2;
#else
	typedef detail::tvec2<mediump_float>	vec2;
#endif//GLM_PRECISION

	namespace precision
	{
		//! 2 components vector of high precision floating-point numbers. 
		//! There is no garanty on the actual precision.
		//! From GLSL 1.30.8 specification, section 4.5.2 Precision Qualifiers.
		typedef detail::tvec2<highp_float>		highp_vec2;
		//! 2 components vector of medium precision floating-point numbers. 
		//! There is no garanty on the actual precision.
		//! From GLSL 1.30.8 specification, section 4.5.2 Precision Qualifiers.
		typedef detail::tvec2<mediump_float>	mediump_vec2;
		//! 2 components vector of low precision floating-point numbers. 
		//! There is no garanty on the actual precision.
		//! From GLSL 1.30.8 specification, section 4.5.2 Precision Qualifiers.
		typedef detail::tvec2<lowp_float>		lowp_vec2;
	}
	//namespace precision

	//////////////////////////
	// Signed integer definition

#ifndef GLM_PRECISION 
	//! \brief 2 components vector of signed integer numbers. 
	//! From GLSL 1.30.8 specification, section 4.1.5 Vectors.
	typedef detail::tvec2<mediump_int>		ivec2;
#elif(GLM_PRECISION & GLM_PRECISION_HIGHP_INT)
	typedef detail::tvec2<highp_int>		ivec2;
#elif(GLM_PRECISION & GLM_PRECISION_MEDIUMP_INT)
	typedef detail::tvec2<mediump_int>		ivec2;
#elif(GLM_PRECISION & GLM_PRECISION_LOWP_INT)
	typedef detail::tvec2<lowp_int>			ivec2;
#else
	typedef detail::tvec2<mediump_int>		ivec2;
#endif//GLM_PRECISION

	namespace precision
	{
		//! 2 components vector of high precision signed integer numbers. 
		//! There is no garanty on the actual precision.
		//! From GLSL 1.30.8 specification, section 4.1.5 Precision Qualifiers.
		typedef detail::tvec2<highp_int>		highp_ivec2;
		//! 2 components vector of medium precision signed integer numbers. 
		//! There is no garanty on the actual precision.
		//! From GLSL 1.30.8 specification, section 4.1.5 Precision Qualifiers.
		typedef detail::tvec2<mediump_int>		mediump_ivec2;
		//! 2 components vector of low precision signed integer numbers.
		//! There is no garanty on the actual precision.
		//! From GLSL 1.30.8 specification, section 4.1.5 Precision Qualifiers.
		typedef detail::tvec2<lowp_int>			lowp_ivec2;
	}
	//namespace precision

	//////////////////////////
	// Unsigned integer definition

#ifndef GLM_PRECISION 
	//! 2 components vector of unsigned integer numbers. 
	//! From GLSL 1.30.8 specification, section 4.1.5 Vectors.
	typedef detail::tvec2<mediump_uint>		uvec2;
#elif(GLM_PRECISION & GLM_PRECISION_HIGHP_UINT)
	typedef detail::tvec2<highp_uint>		uvec2;
#elif(GLM_PRECISION & GLM_PRECISION_MEDIUMP_UINT)
	typedef detail::tvec2<mediump_uint>		uvec2;
#elif(GLM_PRECISION & GLM_PRECISION_LOWP_UINT)
	typedef detail::tvec2<lowp_uint>		uvec2;
#else
	typedef detail::tvec2<mediump_uint>		uvec2;
#endif//GLM_PRECISION

	namespace precision
	{
		//! 2 components vector of high precision unsigned integer numbers. 
		//! There is no garanty on the actual precision.
		//! From GLSL 1.30.8 specification, section 4.1.5 Precision Qualifiers.
		typedef detail::tvec2<highp_uint>		highp_uvec2;
		//! 2 components vector of medium precision unsigned integer numbers. 
		//! There is no garanty on the actual precision.
		//! From GLSL 1.30.8 specification, section 4.1.5 Precision Qualifiers.
		typedef detail::tvec2<mediump_uint>		mediump_uvec2;
		//! 2 components vector of low precision unsigned integer numbers.
		//! There is no garanty on the actual precision.
		//! From GLSL 1.30.8 specification, section 4.1.5 Precision Qualifiers.
		typedef detail::tvec2<lowp_uint>		lowp_uvec2;
	}
	//namespace precision

	}//namespace vector
	}//namespace type
	}//namespace core
}//namespace glm

#include "type_vec2.inl"

#endif//glm_core_type_gentype2
