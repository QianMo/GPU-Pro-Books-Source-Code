///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2008-08-22
// Updated : 2008-08-31
// Licence : This source is under MIT License
// File    : glm/core/type_tvec4.h
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_core_type_gentype4
#define glm_core_type_gentype4

#include "type_float.hpp"
#include "type_int.hpp"
#include "type_size.hpp"
#include "_swizzle.hpp"
#include "_detail.hpp"

namespace glm
{
	namespace test
	{
		void main_vec4();
	}//namespace test

	namespace detail
	{
		template <typename T> struct tref2;
		template <typename T> struct tref3;
		template <typename T> struct tref4;
		template <typename T> struct tvec2;
		template <typename T> struct tvec3;

		template <typename valType>
		struct tvec4
		{
			enum ctor{null};

			//////////////////////////////////////
			// Typedef (Implementation details)

			typedef valType value_type;
			typedef valType & value_reference;
			typedef valType * value_pointer;
			typedef tvec4<bool> bool_type;

			typedef glm::sizeType size_type;
			static size_type value_size();
			static bool is_vector();

			typedef tvec4<valType> type;
			typedef tvec4<valType> * pointer;
			typedef tvec4<valType> const * const_pointer;
			typedef tvec4<valType> const * const const_pointer_const;
			typedef tvec4<valType> * const pointer_const;
			typedef tvec4<valType> & reference;
			typedef tvec4<valType> const & const_reference;
			typedef tvec4<valType> const & param_type;

			//////////////////////////////////////
			// Data

#	if defined(GLM_USE_ONLY_XYZW)
			value_type x, y, z, w;
#	else//GLM_USE_ONLY_XYZW
#		ifdef GLM_USE_ANONYMOUS_UNION
			union 
			{
				struct{value_type x, y, z, w;};
				struct{value_type r, g, b, a;};
				struct{value_type s, t, p, q;};

			};
#		else//GLM_USE_ANONYMOUS_UNION
			union {value_type x, r, s;};
			union {value_type y, g, t;};
			union {value_type z, b, p;};
			union {value_type w, a, q;};
#		endif//GLM_USE_ANONYMOUS_UNION
#	endif//GLM_USE_ONLY_XYZW

			//////////////////////////////////////
			// Accesses

			valType & operator[](size_type i);
			valType const & operator[](size_type i) const;

			//////////////////////////////////////
			// Address (Implementation details)

			value_type * _address(){return (value_type*)(this);}
			value_type const * _address() const{return (value_type*)(this);}

			//////////////////////////////////////
			// Implicit basic constructors

			tvec4();
			tvec4(typename tvec4<valType>::ctor);
			tvec4(tvec4<valType> const & v);

			//////////////////////////////////////
			// Explicit basic constructors

			explicit tvec4(valType const & s);
			explicit tvec4(valType const & s0, valType const & s1, valType const & s2, valType const & s3);

			//////////////////////////////////////
			// Swizzle constructors

			tvec4(tref4<valType> const & r);

			//////////////////////////////////////
			// Convertion scalar constructors

			//! Explicit converions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename valTypeU> 
			explicit tvec4(valTypeU const & x);
			//! Explicit converions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename A, typename B, typename C, typename D> 
			explicit tvec4(A const & x, B const & y, C const & z, D const & w);			

			//////////////////////////////////////
			// Convertion vector constructors

			//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename A, typename B, typename C> 
			explicit tvec4(tvec2<A> const & v, B const & s1, C const & s2);
			//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename A, typename B, typename C> 
			explicit tvec4(A const & s1, tvec2<B> const & v, C const & s2);
			//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename A, typename B, typename C> 
			explicit tvec4(A const & s1, B const & s2, tvec2<C> const & v);
			//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename A, typename B> 
			explicit tvec4(tvec3<A> const & v, B const & s);
			//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename A, typename B> 
			explicit tvec4(A const & s, tvec3<B> const & v);
			//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename A, typename B> 
			explicit tvec4(tvec2<A> const & v1, tvec2<B> const & v2);
			//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename U> 
			explicit tvec4(tvec4<U> const & v);

			//////////////////////////////////////
			// Unary arithmetic operators

			tvec4<valType>& operator= (tvec4<valType> const & v);

			tvec4<valType>& operator+=(valType const & s);
			tvec4<valType>& operator+=(tvec4<valType> const & v);
			tvec4<valType>& operator-=(valType const & s);
			tvec4<valType>& operator-=(tvec4<valType> const & v);
			tvec4<valType>& operator*=(valType const & s);
			tvec4<valType>& operator*=(tvec4<valType> const & v);
			tvec4<valType>& operator/=(valType const & s);
			tvec4<valType>& operator/=(tvec4<valType> const & v);
			tvec4<valType>& operator++();
			tvec4<valType>& operator--();

			//////////////////////////////////////
			// Unary bit operators

			tvec4<valType>& operator%= (valType const & s);
			tvec4<valType>& operator%= (tvec4<valType> const & v);
			tvec4<valType>& operator&= (valType const & s);
			tvec4<valType>& operator&= (tvec4<valType> const & v);
			tvec4<valType>& operator|= (valType const & s);
			tvec4<valType>& operator|= (tvec4<valType> const & v);
			tvec4<valType>& operator^= (valType const & s);
			tvec4<valType>& operator^= (tvec4<valType> const & v);
			tvec4<valType>& operator<<=(valType const & s);
			tvec4<valType>& operator<<=(tvec4<valType> const & v);
			tvec4<valType>& operator>>=(valType const & s);
			tvec4<valType>& operator>>=(tvec4<valType> const & v);

			//////////////////////////////////////
			// Swizzle operators

			valType swizzle(comp X) const;
			tvec2<valType> swizzle(comp X, comp Y) const;
			tvec3<valType> swizzle(comp X, comp Y, comp Z) const;
			tvec4<valType> swizzle(comp X, comp Y, comp Z, comp W) const;
			tref4<valType> swizzle(comp X, comp Y, comp Z, comp W);
		};

		template <typename valType>
		struct tref4
		{
			tref4(valType & x, valType & y, valType & z, valType & w);
			tref4(tref4<valType> const & r);
			tref4(tvec4<valType> const & v);

			tref4<valType>& operator= (tref4<valType> const & r);
			tref4<valType>& operator= (tvec4<valType> const & v);

			valType & x;
			valType & y;
			valType & z;
			valType & w;
		};
	} //namespace detail

	namespace core{
	namespace type{
	namespace vector{

	//////////////////////////
	// Boolean definition

	//! 4 components vector of boolean. 
	//! From GLSL 1.30.8 specification, section 4.1.5 Vectors.
	typedef detail::tvec4<bool>				bvec4;

	//////////////////////////
	// Float definition

#ifndef GLM_PRECISION 
	//! 4 components vector of floating-point numbers. 
	//! From GLSL 1.30.8 specification, section 4.1.5 Vectors.
	typedef detail::tvec4<mediump_float>	vec4;
#elif(GLM_PRECISION & GLM_PRECISION_HIGHP_FLOAT)
	typedef detail::tvec4<highp_float>		vec4;
#elif(GLM_PRECISION & GLM_PRECISION_MEDIUMP_FLOAT)
	typedef detail::tvec4<mediump_float>	vec4;
#elif(GLM_PRECISION & GLM_PRECISION_LOWP_FLOAT)
	typedef detail::tvec4<lowp_float>		vec4;
#endif//GLM_PRECISION

	namespace precision
	{
		//! 4 components vector of high precision floating-point numbers. 
		//! There is no garanty on the actual precision.
		//! From GLSL 1.30.8 specification, section 4.5.2 Precision Qualifiers.
		typedef detail::tvec4<highp_float>		highp_vec4;
		//! 4 components vector of medium precision floating-point numbers. 
		//! There is no garanty on the actual precision.
		//! From GLSL 1.30.8 specification, section 4.5.2 Precision Qualifiers.
		typedef detail::tvec4<mediump_float>	mediump_vec4;
		//! 4 components vector of low precision floating-point numbers. 
		//! There is no garanty on the actual precision.
		//! From GLSL 1.30.8 specification, section 4.5.2 Precision Qualifiers.
		typedef detail::tvec4<lowp_float>		lowp_vec4;
	}
	//namespace precision

	//////////////////////////
	// Signed integer definition

#ifndef GLM_PRECISION 
	//! 4 components vector of signed integer numbers. 
	//! From GLSL 1.30.8 specification, section 4.1.5 Vectors.
	typedef detail::tvec4<mediump_int>		ivec4;
#elif(GLM_PRECISION & GLM_PRECISION_HIGHP_INT)
	typedef detail::tvec4<highp_int>		ivec4;
#elif(GLM_PRECISION & GLM_PRECISION_MEDIUMP_INT)
	typedef detail::tvec4<mediump_int>		ivec4;
#elif(GLM_PRECISION & GLM_PRECISION_LOWP_INT)
	typedef detail::tvec4<lowp_int>			ivec4;
#endif//GLM_PRECISION

	namespace precision
	{
		//! 4 components vector of high precision signed integer numbers. 
		//! There is no garanty on the actual precision.
		//! From GLSL 1.30.8 specification, section 4.1.5 Precision Qualifiers.
		typedef detail::tvec4<highp_int>		highp_ivec4;
		//! 4 components vector of medium precision signed integer numbers. 
		//! There is no garanty on the actual precision.
		//! From GLSL 1.30.8 specification, section 4.1.5 Precision Qualifiers.
		typedef detail::tvec4<mediump_int>		mediump_ivec4;
		//! 4 components vector of low precision signed integer numbers. 
		//! There is no garanty on the actual precision.
		//! From GLSL 1.30.8 specification, section 4.1.5 Precision Qualifiers.
		typedef detail::tvec4<lowp_int>			lowp_ivec4;
	}
	//namespace precision

	//////////////////////////
	// Unsigned integer definition

#ifndef GLM_PRECISION 
	//! 4 components vector of unsigned integer numbers. 
	//! From GLSL 1.30.8 specification, section 4.1.5 Vectors.
	typedef detail::tvec4<mediump_uint>		uvec4;
#elif(GLM_PRECISION & GLM_PRECISION_HIGHP_UINT)
	typedef detail::tvec4<highp_uint>		uvec4;
#elif(GLM_PRECISION & GLM_PRECISION_MEDIUMP_UINT)
	typedef detail::tvec4<mediump_uint>		uvec4;
#elif(GLM_PRECISION & GLM_PRECISION_LOWP_UINT)
	typedef detail::tvec4<lowp_uint>		uvec4;
#endif//GLM_PRECISION

	namespace precision
	{
		//! 4 components vector of high precision unsigned integer numbers. 
		//! There is no garanty on the actual precision.
		//! From GLSL 1.30.8 specification, section 4.1.5 Precision Qualifiers.
		typedef detail::tvec4<highp_uint>		highp_uvec4;
		//! 4 components vector of medium precision unsigned integer numbers. 
		//! There is no garanty on the actual precision.
		//! From GLSL 1.30.8 specification, section 4.1.5 Precision Qualifiers.
		typedef detail::tvec4<mediump_uint>		mediump_uvec4;
		//! 4 components vector of low precision unsigned integer numbers. 
		//! There is no garanty on the actual precision.
		//! From GLSL 1.30.8 specification, section 4.1.5 Precision Qualifiers.
		typedef detail::tvec4<lowp_uint>		lowp_uvec4;
	}
	//namespace precision

	}//namespace vector
	}//namespace type
	}//namespace core

	using namespace core::type;

}//namespace glm

#include "type_vec4.inl"

#endif//glm_core_type_gentype4
