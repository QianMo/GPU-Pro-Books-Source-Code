///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2006-08-04
// Updated : 2006-08-30
// Licence : This source is under MIT License
// File    : glm/core/type_type_mat4x3.h
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_core_type_mat4x3
#define glm_core_type_mat4x3

#include "type_size.hpp"

namespace glm
{
	namespace test
	{
		void main_mat4x3();
	}//namespace test

	namespace detail
	{
		template <typename T> struct tvec1;
		template <typename T> struct tvec2;
		template <typename T> struct tvec3;
		template <typename T> struct tvec4;
		template <typename T> struct tmat2x2;
		template <typename T> struct tmat2x3;
		template <typename T> struct tmat2x4;
		template <typename T> struct tmat3x2;
		template <typename T> struct tmat3x3;
		template <typename T> struct tmat3x4;
		template <typename T> struct tmat4x2;
		template <typename T> struct tmat4x3;
		template <typename T> struct tmat4x4;

		//!< \brief Template for 4 * 3 matrix of floating-point numbers.
		template <typename T> 
		struct tmat4x3
		{
		public:
			typedef tmat4x3<T>* pointer;
			typedef const tmat4x3<T>* const_pointer;
			typedef const tmat4x3<T>*const const_pointer_const;
			typedef tmat4x3<T>*const pointer_const;
			typedef tmat4x3<T>& reference;
			typedef const tmat4x3<T>& const_reference;
			typedef const tmat4x3<T>& param_type;
			typedef tmat3x4<T> transpose_type;

			typedef T value_type;
			typedef detail::tvec4<T> col_type;
			typedef detail::tvec3<T> row_type;
			typedef glm::sizeType size_type;
			static size_type col_size();
			static size_type row_size();
			static bool is_matrix();

		public:
			tmat3x4<T> _inverse() const;

		private:
			// Data 
			detail::tvec3<T> value[4];

		public:
			// Constructors
			tmat4x3();
			explicit tmat4x3(T const & x);
			explicit tmat4x3(
				const T x0, const T y0, const T z0,
				const T x1, const T y1, const T z1,
				const T x2, const T y2, const T z2,
				const T x3, const T y3, const T z3);
			explicit tmat4x3(
				const detail::tvec3<T>& v0, 
				const detail::tvec3<T>& v1,
				const detail::tvec3<T>& v2,
				const detail::tvec3<T>& v3);

			// Conversion
			template <typename U> 
			explicit tmat4x3(const tmat4x3<U>& m);
			
			explicit tmat4x3(const tmat2x2<T>& x);
			explicit tmat4x3(const tmat3x3<T>& x);
			explicit tmat4x3(const tmat4x4<T>& x);
			explicit tmat4x3(const tmat2x3<T>& x);
			explicit tmat4x3(const tmat3x2<T>& x);
			explicit tmat4x3(const tmat2x4<T>& x);
			explicit tmat4x3(const tmat4x2<T>& x);
			explicit tmat4x3(const tmat3x4<T>& x);

			// Accesses
			detail::tvec3<T>& operator[](size_type i);
			detail::tvec3<T> const & operator[](size_type i) const;

			// Unary updatable operators
			tmat4x3<T>& operator=  (tmat4x3<T> const & m);
			tmat4x3<T>& operator+= (T const & s);
			tmat4x3<T>& operator+= (tmat4x3<T> const & m);
			tmat4x3<T>& operator-= (T const & s);
			tmat4x3<T>& operator-= (tmat4x3<T> const & m);
			tmat4x3<T>& operator*= (T const & s);
			tmat4x3<T>& operator*= (tmat3x4<T> const & m);
			tmat4x3<T>& operator/= (T const & s);
			//tmat4x3<T>& operator/= (tmat3x4<T> const & m);

			tmat4x3<T>& operator++ ();
			tmat4x3<T>& operator-- ();

			// Unary constant operators
			const tmat4x3<T> operator- () const;
			const tmat4x3<T> operator++ (int) const;
			const tmat4x3<T> operator-- (int) const;
		};

		// Binary operators
		template <typename T> 
		tmat4x3<T> operator+ (const tmat4x3<T>& m, const T & s);
	    
		template <typename T> 
		tmat4x3<T> operator+ (const tmat4x3<T>& m1, const tmat4x3<T>& m2);
	    
		template <typename T> 
		tmat4x3<T> operator- (const tmat4x3<T>& m, const T & s);

		template <typename T> 
		tmat4x3<T> operator- (const tmat4x3<T>& m1, const tmat4x3<T>& m2);

		template <typename T> 
		tmat4x3<T> operator* (const tmat4x3<T>& m, const T & s);

		template <typename T> 
		tmat4x3<T> operator* (const T & s, const tmat4x3<T>& m);

		template <typename T>
		detail::tvec3<T> operator* (const tmat4x3<T>& m, const detail::tvec4<T>& v);

		template <typename T> 
		detail::tvec4<T> operator* (const detail::tvec3<T>& v, const tmat4x3<T>& m);

		template <typename T> 
		tmat3x3<T> operator* (const tmat4x3<T>& m1, const tmat3x4<T>& m2);

		template <typename T> 
		tmat4x3<T> operator/ (const tmat4x3<T>& m, const T & s);

		template <typename T> 
		tmat4x3<T> operator/ (const T & s, const tmat4x3<T> & m);

		//template <typename T> 
		//detail::tvec3<T> operator/ (const tmat4x3<T>& m, const detail::tvec4<T>& v);

		//template <typename T> 
		//detail::tvec4<T> operator/ (const detail::tvec3<T>& v, const tmat4x3<T>& m);

		//template <typename T> 
		//tmat3x3<T> operator/ (const tmat4x3<T>& m1, const tmat3x4<T>& m2);

		// Unary constant operators
		template <typename valType> 
		tmat4x3<valType> const operator- (tmat4x3<valType> const & m);

		template <typename valType> 
		tmat4x3<valType> const operator-- (tmat4x3<valType> const & m, int);

		template <typename valType> 
		tmat4x3<valType> const operator++ (tmat4x3<valType> const & m, int);

	} //namespace detail

	namespace core{
	namespace type{
	namespace matrix{

	namespace precision
	{
		//! 4 columns of 3 components matrix of low precision floating-point numbers.
		//! There is no garanty on the actual precision.
		//! (From GLSL 1.30.8 specification, section 4.1.6 Matrices and section 4.5 Precision and Precision Qualifiers)
		typedef detail::tmat4x3<lowp_float>		lowp_mat4x3;
		//! 4 columns of 3 components matrix of medium precision floating-point numbers.
		//! There is no garanty on the actual precision.
		//! (From GLSL 1.30.8 specification, section 4.1.6 Matrices and section 4.5 Precision and Precision Qualifiers)
		typedef detail::tmat4x3<mediump_float>	mediump_mat4x3;
		//! 4 columns of 3 components matrix of high precision floating-point numbers.
		//! There is no garanty on the actual precision.
		//! (From GLSL 1.30.8 specification, section 4.1.6 Matrices and section 4.5 Precision and Precision Qualifiers)
		typedef detail::tmat4x3<highp_float>	highp_mat4x3;
	}
	//namespace precision

#ifndef GLM_PRECISION
	//! 4 columns of 3 components matrix of floating-point numbers. 
	//! (From GLSL 1.30.8 specification, section 4.1.6 Matrices)
	typedef detail::tmat4x3<mediump_float>	mat4x3;
#elif(GLM_PRECISION & GLM_PRECISION_HIGH)
	typedef detail::tmat4x3<highp_float>	mat4x3;
#elif(GLM_PRECISION & GLM_PRECISION_MEDIUM)
	typedef detail::tmat4x3<mediump_float>	mat4x3;
#elif(GLM_PRECISION & GLM_PRECISION_LOW)
	typedef detail::tmat4x3<lowp_float>		mat4x3;
#else
	typedef detail::tmat4x3<mediump_float>	mat4x3;
#endif//GLM_PRECISION

	}//namespace matrix
	}//namespace type
	}//namespace core
} //namespace glm

#include "type_mat4x3.inl"

#endif//glm_core_type_mat4x3
