///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2009-04-29
// Updated : 2009-04-29
// Licence : This source is under MIT License
// File    : glm/gtc/half_float.hpp
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_gtc_half_float
#define glm_gtc_half_float

// Dependency:
#include "../glm.hpp"

namespace glm
{
	namespace test{
		bool main_gtc_half_float();
	}//namespace test

	namespace detail
	{
#ifndef GLM_USE_ANONYMOUS_UNION
		template <>
		struct tvec2<thalf>
		{
			//////////////////////////////////////
			// Typedef (Implementation details)

			typedef thalf value_type;
			typedef thalf & value_reference;
			typedef thalf * value_pointer;
			typedef tvec2<bool> bool_type;

			typedef sizeType size_type;
			static size_type value_size();
			static bool is_vector();

			typedef tvec2<thalf> type;
			typedef tvec2<thalf>* pointer;
			typedef const tvec2<thalf>* const_pointer;
			typedef const tvec2<thalf>*const const_pointer_const;
			typedef tvec2<thalf>*const pointer_const;
			typedef tvec2<thalf>& reference;
			typedef const tvec2<thalf>& const_reference;
			typedef const tvec2<thalf>& param_type;

			//////////////////////////////////////
			// Data

			thalf x, y;

			//////////////////////////////////////
			// Accesses

			thalf & operator[](size_type i);
			thalf const & operator[](size_type i) const;

			//////////////////////////////////////
			// Address (Implementation details)

			thalf const * _address() const{return (value_type*)(this);}
			thalf * _address(){return (value_type*)(this);}

			//////////////////////////////////////
			// Implicit basic constructors

			tvec2();
			tvec2(tvec2<thalf> const & v);

			//////////////////////////////////////
			// Explicit basic constructors

			explicit tvec2(thalf s);
			explicit tvec2(thalf s1, thalf s2);

			//////////////////////////////////////
			// Swizzle constructors

			tvec2(tref2<thalf> const & r);

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

			tvec2<thalf>& operator= (tvec2<thalf> const & v);

			tvec2<thalf>& operator+=(thalf s);
			tvec2<thalf>& operator+=(tvec2<thalf> const & v);
			tvec2<thalf>& operator-=(thalf s);
			tvec2<thalf>& operator-=(tvec2<thalf> const & v);
			tvec2<thalf>& operator*=(thalf s);
			tvec2<thalf>& operator*=(tvec2<thalf> const & v);
			tvec2<thalf>& operator/=(thalf s);
			tvec2<thalf>& operator/=(tvec2<thalf> const & v);
			tvec2<thalf>& operator++();
			tvec2<thalf>& operator--();

			//////////////////////////////////////
			// Swizzle operators

			thalf swizzle(comp X) const;
			tvec2<thalf> swizzle(comp X, comp Y) const;
			tvec3<thalf> swizzle(comp X, comp Y, comp Z) const;
			tvec4<thalf> swizzle(comp X, comp Y, comp Z, comp W) const;
			tref2<thalf> swizzle(comp X, comp Y);
		};

		template <>
		struct tvec3<thalf>
		{
			//////////////////////////////////////
			// Typedef (Implementation details)

			typedef thalf value_type;
			typedef thalf & value_reference;
			typedef thalf * value_pointer;
			typedef tvec3<bool> bool_type;

			typedef glm::sizeType size_type;
			static size_type value_size();
			static bool is_vector();

			typedef tvec3<thalf> type;
			typedef tvec3<thalf> * pointer;
			typedef tvec3<thalf> const * const_pointer;
			typedef tvec3<thalf> const * const const_pointer_const;
			typedef tvec3<thalf> * const pointer_const;
			typedef tvec3<thalf> & reference;
			typedef tvec3<thalf> const & const_reference;
			typedef tvec3<thalf> const & param_type;

			//////////////////////////////////////
			// Data

			thalf x, y, z;

			//////////////////////////////////////
			// Accesses

			thalf & operator[](size_type i);
			thalf const & operator[](size_type i) const;

			//////////////////////////////////////
			// Address (Implementation details)

			value_type const * _address() const{return (value_type*)(this);}
			value_type * _address(){return (value_type*)(this);}

			//////////////////////////////////////
			// Implicit basic constructors

			tvec3();
			tvec3(tvec3<thalf> const & v);

			//////////////////////////////////////
			// Explicit basic constructors

			explicit tvec3(thalf s);
			explicit tvec3(thalf s1, thalf s2, thalf s3);

			//////////////////////////////////////
			// Swizzle constructors

			tvec3(tref3<thalf> const & r);

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
			explicit tvec3(const tvec2<A>& v, B s);
			//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename A, typename B> 
			explicit tvec3(A s, const tvec2<B>& v);
			//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename U> 
			explicit tvec3(tvec3<U> const & v);
			//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename U> 
			explicit tvec3(tvec4<U> const & v);

			//////////////////////////////////////
			// Unary arithmetic operators

			tvec3<thalf>& operator= (tvec3<thalf> const & v);

			tvec3<thalf>& operator+=(thalf s);
			tvec3<thalf>& operator+=(tvec3<thalf> const & v);
			tvec3<thalf>& operator-=(thalf s);
			tvec3<thalf>& operator-=(tvec3<thalf> const & v);
			tvec3<thalf>& operator*=(thalf s);
			tvec3<thalf>& operator*=(tvec3<thalf> const & v);
			tvec3<thalf>& operator/=(thalf s);
			tvec3<thalf>& operator/=(tvec3<thalf> const & v);
			tvec3<thalf>& operator++();
			tvec3<thalf>& operator--();

			//////////////////////////////////////
			// Swizzle operators

			thalf swizzle(comp X) const;
			tvec2<thalf> swizzle(comp X, comp Y) const;
			tvec3<thalf> swizzle(comp X, comp Y, comp Z) const;
			tvec4<thalf> swizzle(comp X, comp Y, comp Z, comp W) const;
			tref3<thalf> swizzle(comp X, comp Y, comp Z);
		};

		template <>
		struct tvec4<thalf>
		{
			//////////////////////////////////////
			// Typedef (Implementation details)

			typedef thalf value_type;
			typedef thalf & value_reference;
			typedef thalf * value_pointer;
			typedef tvec4<bool> bool_type;

			typedef glm::sizeType size_type;
			static size_type value_size();
			static bool is_vector();

			typedef tvec4<thalf> type;
			typedef tvec4<thalf> * pointer;
			typedef tvec4<thalf> const * const_pointer;
			typedef tvec4<thalf> const * const const_pointer_const;
			typedef tvec4<thalf> * const pointer_const;
			typedef tvec4<thalf> & reference;
			typedef tvec4<thalf> const & const_reference;
			typedef tvec4<thalf> const & param_type;

			//////////////////////////////////////
			// Data

			thalf x, y, z, w;

			//////////////////////////////////////
			// Accesses

			thalf & operator[](size_type i);
			thalf const & operator[](size_type i) const;

			//////////////////////////////////////
			// Address (Implementation details)

			value_type const * _address() const{return (value_type*)(this);}
			value_type * _address(){return (value_type*)(this);}

			//////////////////////////////////////
			// Implicit basic constructors

			tvec4();
			tvec4(tvec4<thalf> const & v);

			//////////////////////////////////////
			// Explicit basic constructors

			explicit tvec4(thalf s);
			explicit tvec4(thalf s0, thalf s1, thalf s2, thalf s3);

			//////////////////////////////////////
			// Swizzle constructors

			tvec4(tref4<thalf> const & r);

			//////////////////////////////////////
			// Convertion scalar constructors

			//! Explicit converions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename U> 
			explicit tvec4(U x);
			//! Explicit converions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename A, typename B, typename C, typename D> 
			explicit tvec4(A x, B y, C z, D w);			

			//////////////////////////////////////
			// Convertion vector constructors

			//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename A, typename B, typename C> 
			explicit tvec4(const tvec2<A>& v, B s1, C s2);
			//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename A, typename B, typename C> 
			explicit tvec4(A s1, const tvec2<B>& v, C s2);
			//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename A, typename B, typename C> 
			explicit tvec4(A s1, B s2, const tvec2<C>& v);
			//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename A, typename B> 
			explicit tvec4(const tvec3<A>& v, B s);
			//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename A, typename B> 
			explicit tvec4(A s, const tvec3<B>& v);
			//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename A, typename B> 
			explicit tvec4(const tvec2<A>& v1, const tvec2<B>& v2);
			//! Explicit conversions (From section 5.4.1 Conversion and scalar constructors of GLSL 1.30.08 specification)
			template <typename U> 
			explicit tvec4(const tvec4<U>& v);

			//////////////////////////////////////
			// Unary arithmetic operators

			tvec4<thalf>& operator= (tvec4<thalf> const & v);

			tvec4<thalf>& operator+=(thalf s);
			tvec4<thalf>& operator+=(tvec4<thalf> const & v);
			tvec4<thalf>& operator-=(thalf s);
			tvec4<thalf>& operator-=(tvec4<thalf> const & v);
			tvec4<thalf>& operator*=(thalf s);
			tvec4<thalf>& operator*=(tvec4<thalf> const & v);
			tvec4<thalf>& operator/=(thalf s);
			tvec4<thalf>& operator/=(tvec4<thalf> const & v);
			tvec4<thalf>& operator++();
			tvec4<thalf>& operator--();

			//////////////////////////////////////
			// Swizzle operators

			thalf swizzle(comp X) const;
			tvec2<thalf> swizzle(comp X, comp Y) const;
			tvec3<thalf> swizzle(comp X, comp Y, comp Z) const;
			tvec4<thalf> swizzle(comp X, comp Y, comp Z, comp W) const;
			tref4<thalf> swizzle(comp X, comp Y, comp Z, comp W);
		};
#endif//GLM_USE_ANONYMOUS_UNION
	}
	//namespace detail

	namespace gtc{
	//! GLM_GTC_half_float extension: Add support for half precision floating-point types
	namespace half_float
	{
		//! Type for half-precision floating-point numbers. 
		//! From GLM_GTC_half_float extension.
		typedef detail::thalf					half;

		//! Vector of 2 half-precision floating-point numbers. 
		//! From GLM_GTC_half_float extension.
		typedef detail::tvec2<detail::thalf>	hvec2;

		//! Vector of 3 half-precision floating-point numbers.
		//! From GLM_GTC_half_float extension.
		typedef detail::tvec3<detail::thalf>	hvec3;

		//! Vector of 4 half-precision floating-point numbers. 
		//! From GLM_GTC_half_float extension.
		typedef detail::tvec4<detail::thalf>	hvec4;

		//! 2 * 2 matrix of half-precision floating-point numbers.
		//! From GLM_GTC_half_float extension.
		typedef detail::tmat2x2<detail::thalf>	hmat2;

		//! 3 * 3 matrix of half-precision floating-point numbers.
		//! From GLM_GTC_half_float extension.
		typedef detail::tmat3x3<detail::thalf>	hmat3;

		//! 4 * 4 matrix of half-precision floating-point numbers.
		//! From GLM_GTC_half_float extension.
		typedef detail::tmat4x4<detail::thalf>	hmat4;

	}//namespace half_float
	}//namespace gtc
}//namespace glm

#define GLM_GTC_half_float namespace gtc::half_float
#ifndef GLM_GTC_GLOBAL
namespace glm {using GLM_GTC_half_float;}
#endif//GLM_GTC_GLOBAL

#include "half_float.inl"

#endif//glm_gtc_half_float
