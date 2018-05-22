///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2008-08-22
// Updated : 2008-08-31
// Licence : This source is under MIT License
// File    : glm/core/type_tvec3.h
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_core_type_gentype3
#define glm_core_type_gentype3

#include "type_float.hpp"
#include "type_int.hpp"
#include "type_size.hpp"
#include "_swizzle.hpp"

namespace glm
{
	namespace test
	{
		void main_vec3();
	}//namespace test

	namespace detail
	{
		template <typename T> struct tref2;
		template <typename T> struct tref3;
		template <typename T> struct tref4;
		template <typename T> struct tvec2;
		template <typename T> struct tvec4;

		template <typename valType>
		struct tvec3
		{
			//////////////////////////////////////
			// Typedef (Implementation details)

			typedef valType value_type;
			typedef valType& value_reference;
			typedef valType* value_pointer;
			typedef tvec3<bool> bool_type;

			typedef glm::sizeType size_type;
			static size_type value_size();
			static bool is_vector();

			typedef tvec3<valType> type;
			typedef tvec3<valType>* pointer;
			typedef tvec3<valType> const * const_pointer;
			typedef tvec3<valType> const * const const_pointer_const;
			typedef tvec3<valType> * const pointer_const;
			typedef tvec3<valType>& reference;
			typedef tvec3<valType> const & const_reference;
			typedef tvec3<valType> const & param_type;

			//////////////////////////////////////
			// Data

#	if defined(GLM_USE_ONLY_XYZW)
			value_type x, y, z;
#	else//GLM_USE_ONLY_XYZW
#		ifdef GLM_USE_ANONYMOUS_UNION
			union 
			{
				struct{value_type x, y, z;};
				struct{value_type r, g, b;};
				struct{value_type s, t, p;};
			};
#		else//GLM_USE_ANONYMOUS_UNION
			union {value_type x, r, s;};
			union {value_type y, g, t;};
			union {value_type z, b, p;};
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

			tvec3();
			tvec3(tvec3<valType> const & v);

			//////////////////////////////////////
			// Explicit basic constructors

			explicit tvec3(valType s);
			explicit tvec3(valType s1, valType s2, valType s3);

			//////////////////////////////////////
			// Swizzle constructors

			tvec3(tref3<valType> const & r);

			//////////////////////////////////////
			// Convertion scalar constructors

			//! Explicit converions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename U> 
			explicit tvec3(U x);
			//! Explicit converions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename U, typename V, typename W> 
			explicit tvec3(U x, V y, W z);			

			//////////////////////////////////////
			// Convertion vector constructors

			//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename A, typename B> 
			explicit tvec3(tvec2<A> const & v, B s);
			//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename A, typename B> 
			explicit tvec3(A s, tvec2<B> const & v);
			//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename U> 
			explicit tvec3(tvec3<U> const & v);
			//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename U> 
			explicit tvec3(tvec4<U> const & v);

			//////////////////////////////////////
			// Unary arithmetic operators

			tvec3<valType>& operator= (tvec3<valType> const & v);

			tvec3<valType>& operator+=(valType const & s);
			tvec3<valType>& operator+=(tvec3<valType> const & v);
			tvec3<valType>& operator-=(valType const & s);
			tvec3<valType>& operator-=(tvec3<valType> const & v);
			tvec3<valType>& operator*=(valType const & s);
			tvec3<valType>& operator*=(tvec3<valType> const & v);
			tvec3<valType>& operator/=(valType const & s);
			tvec3<valType>& operator/=(tvec3<valType> const & v);
			tvec3<valType>& operator++();
			tvec3<valType>& operator--();

			//////////////////////////////////////
			// Unary bit operators

			tvec3<valType>& operator%=(valType const & s);
			tvec3<valType>& operator%=(tvec3<valType> const & v);
			tvec3<valType>& operator&=(valType const & s);
			tvec3<valType>& operator&=(tvec3<valType> const & v);
			tvec3<valType>& operator|=(valType const & s);
			tvec3<valType>& operator|=(tvec3<valType> const & v);
			tvec3<valType>& operator^=(valType const & s);
			tvec3<valType>& operator^=(tvec3<valType> const & v);
			tvec3<valType>& operator<<=(valType const & s);
			tvec3<valType>& operator<<=(tvec3<valType> const & v);
			tvec3<valType>& operator>>=(valType const & s);
			tvec3<valType>& operator>>=(tvec3<valType> const & v);

			//////////////////////////////////////
			// Swizzle operators

			valType swizzle(comp X) const;
			tvec2<valType> swizzle(comp X, comp Y) const;
			tvec3<valType> swizzle(comp X, comp Y, comp Z) const;
			tvec4<valType> swizzle(comp X, comp Y, comp Z, comp W) const;
			tref3<valType> swizzle(comp X, comp Y, comp Z);
		};

		template <typename T>
		struct tref3
		{
			tref3(T& x, T& y, T& z);
			tref3(tref3<T> const & r);
			tref3(tvec3<T> const & v);

			tref3<T>& operator= (tref3<T> const & r);
			tref3<T>& operator= (tvec3<T> const & v);

			T& x;
			T& y;
			T& z;
		};
	} //namespace detail

	namespace core{
	namespace type{
	namespace vector{

	//////////////////////////
	// Boolean definition

	//! 3 components vector of boolean. 
	//! From GLSL 1.30.8 specification, section 4.1.5 Vectors.
	typedef detail::tvec3<bool>				bvec3;

	//////////////////////////
	// Float definition

#ifndef GLM_PRECISION 
	//! 3 components vector of floating-point numbers. 
	//! From GLSL 1.30.8 specification, section 4.1.5 Vectors.
	typedef detail::tvec3<mediump_float>	vec3;
#elif(GLM_PRECISION & GLM_PRECISION_HIGHP_FLOAT)
	typedef detail::tvec3<highp_float>		vec3;
#elif(GLM_PRECISION & GLM_PRECISION_MEDIUMP_FLOAT)
	typedef detail::tvec3<mediump_float>	vec3;
#elif(GLM_PRECISION & GLM_PRECISION_LOWP_FLOAT)
	typedef detail::tvec3<lowp_float>		vec3;
#else
	typedef detail::tvec3<mediump_float>	vec3;
#endif//GLM_PRECISION

	namespace precision
	{
		//! 3 components vector of high precision floating-point numbers. 
		//! There is no garanty on the actual precision.
		//! From GLSL 1.30.8 specification, section 4.5.2 Precision Qualifiers.
		typedef detail::tvec3<highp_float>		highp_vec3;
		//! 3 components vector of medium precision floating-point numbers. 
		//! There is no garanty on the actual precision.
		//! From GLSL 1.30.8 specification, section 4.5.2 Precision Qualifiers.
		typedef detail::tvec3<mediump_float>	mediump_vec3;
		//! 3 components vector of low precision floating-point numbers. 
		//! There is no garanty on the actual precision.
		//! From GLSL 1.30.8 specification, section 4.5.2 Precision Qualifiers.
		typedef detail::tvec3<lowp_float>		lowp_vec3;
	}
	//namespace precision

	//////////////////////////
	// Signed integer definition

#ifndef GLM_PRECISION 
	//! 3 components vector of signed integer numbers. 
	//! From GLSL 1.30.8 specification, section 4.1.5 Vectors.
	typedef detail::tvec3<mediump_int>		ivec3;
#elif(GLM_PRECISION & GLM_PRECISION_HIGHP_INT)
	typedef detail::tvec3<highp_int>		ivec3;
#elif(GLM_PRECISION & GLM_PRECISION_MEDIUMP_INT)
	typedef detail::tvec3<mediump_int>		ivec3;
#elif(GLM_PRECISION & GLM_PRECISION_LOWP_INT)
	typedef detail::tvec3<lowp_int>			ivec3;
#else
	typedef detail::tvec3<mediump_int>		ivec3;
#endif//GLM_PRECISION

	namespace precision
	{
		//! 3 components vector of high precision signed integer numbers. 
		//! There is no garanty on the actual precision.
		//! From GLSL 1.30.8 specification, section 4.1.5 Precision Qualifiers.
		typedef detail::tvec3<highp_int>		highp_ivec3;
		//! 3 components vector of medium precision signed integer numbers. 
		//! There is no garanty on the actual precision.
		//! From GLSL 1.30.8 specification, section 4.1.5 Precision Qualifiers.
		typedef detail::tvec3<mediump_int>		mediump_ivec3;
		//! 3 components vector of low precision signed integer numbers. 
		//! There is no garanty on the actual precision.
		//! From GLSL 1.30.8 specification, section 4.1.5 Precision Qualifiers.
		typedef detail::tvec3<lowp_int>			lowp_ivec3;
	}
	//namespace precision

	//////////////////////////
	// Unsigned integer definition

#ifndef GLM_PRECISION 
	//! 3 components vector of unsigned integer numbers. 
	//! From GLSL 1.30.8 specification, section 4.1.5 Vectors.
	typedef detail::tvec3<mediump_uint>		uvec3;
#elif(GLM_PRECISION & GLM_PRECISION_HIGHP_UINT)
	typedef detail::tvec3<highp_uint>		uvec3;
#elif(GLM_PRECISION & GLM_PRECISION_MEDIUMP_UINT)
	typedef detail::tvec3<mediump_uint>		uvec3;
#elif(GLM_PRECISION & GLM_PRECISION_LOWP_UINT)
	typedef detail::tvec3<lowp_uint>		uvec3;
#else
	typedef detail::tvec3<mediump_uint>		uvec3;
#endif//GLM_PRECISION

	namespace precision
	{
		//! 3 components vector of high precision unsigned integer numbers. 
		//! There is no garanty on the actual precision.
		//! From GLSL 1.30.8 specification, section 4.1.5 Precision Qualifiers.
		typedef detail::tvec3<highp_uint>		highp_uvec3;
		//! 3 components vector of medium precision unsigned integer numbers. 
		//! There is no garanty on the actual precision.
		//! From GLSL 1.30.8 specification, section 4.1.5 Precision Qualifiers.
		typedef detail::tvec3<mediump_uint>		mediump_uvec3;
		//! 3 components vector of low precision unsigned integer numbers. 
		//! There is no garanty on the actual precision.
		//! From GLSL 1.30.8 specification, section 4.1.5 Precision Qualifiers.
		typedef detail::tvec3<lowp_uint>		lowp_uvec3;
	}
	//namespace precision
		
	}//namespace vector
	}//namespace type
	}//namespace core
}//namespace glm

#include "type_vec3.inl"

#endif//glm_core_type_gentype3
