///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2005-01-27
// Updated : 2008-08-30
// Licence : This source is under MIT License
// File    : glm/core/type_mat4x4.h
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_core_type_mat4x4
#define glm_core_type_mat4x4

#include "type_size.hpp"

namespace glm
{
	namespace test
	{
		void main_mat4x4();
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

		//!< \brief Template for 4 * 4 matrix of floating-point numbers.
		template <typename T> 
		struct tmat4x4
		{
		public:
			enum ctor{null};

			typedef tmat4x4<T>* pointer;
			typedef const tmat4x4<T>* const_pointer;
			typedef const tmat4x4<T>*const const_pointer_const;
			typedef tmat4x4<T>*const pointer_const;
			typedef tmat4x4<T>& reference;
			typedef const tmat4x4<T>& const_reference;
			typedef const tmat4x4<T>& param_type;
			typedef tmat4x4<T> transpose_type;

			typedef T value_type;
			typedef tvec4<T> col_type;
			typedef tvec4<T> row_type;
			typedef glm::sizeType size_type;
			static size_type value_size();
			static size_type col_size();
			static size_type row_size();
			static bool is_matrix();

		public:
			tmat4x4<T> _inverse() const;

		private:
			// Data 
			detail::tvec4<T> value[4];

		public:
			// Constructors
			tmat4x4();
			explicit tmat4x4(ctor Null);
			explicit tmat4x4(T const & x);
			explicit tmat4x4(
				const T x0, const T y0, const T z0, const T w0,
				const T x1, const T y1, const T z1, const T w1,
				const T x2, const T y2, const T z2, const T w2,
				const T x3, const T y3, const T z3, const T w3);
			explicit tmat4x4(
				detail::tvec4<T> const & v0, 
				detail::tvec4<T> const & v1,
				detail::tvec4<T> const & v2,
				detail::tvec4<T> const & v3);

			// Conversions
			template <typename U> 
			explicit tmat4x4(tmat4x4<U> const & m);

			explicit tmat4x4(tmat2x2<T> const & x);
			explicit tmat4x4(tmat3x3<T> const & x);
			explicit tmat4x4(tmat2x3<T> const & x);
			explicit tmat4x4(tmat3x2<T> const & x);
			explicit tmat4x4(tmat2x4<T> const & x);
			explicit tmat4x4(tmat4x2<T> const & x);
			explicit tmat4x4(tmat3x4<T> const & x);
			explicit tmat4x4(tmat4x3<T> const & x);

			// Accesses
			detail::tvec4<T>& operator[](size_type i);
			detail::tvec4<T> const & operator[](size_type i) const;

			// Unary updatable operators
			tmat4x4<T>& operator= (tmat4x4<T> const & m);
			tmat4x4<T>& operator+= (T const & s);
			tmat4x4<T>& operator+= (tmat4x4<T> const & m);
			tmat4x4<T>& operator-= (T const & s);
			tmat4x4<T>& operator-= (tmat4x4<T> const & m);
			tmat4x4<T>& operator*= (T const & s);
			tmat4x4<T>& operator*= (tmat4x4<T> const & m);
			tmat4x4<T>& operator/= (T const & s);
			tmat4x4<T>& operator/= (tmat4x4<T> const & m);
			tmat4x4<T>& operator++ ();
			tmat4x4<T>& operator-- ();
		};

		// Binary operators
		template <typename valType> 
		tmat4x4<valType> operator+ (const tmat4x4<valType>& m, valType const & s);

		template <typename valType> 
		tmat4x4<valType> operator+ (valType const & s, const tmat4x4<valType>& m);

		template <typename T> 
		tmat4x4<T> operator+ (const tmat4x4<T>& m1, const tmat4x4<T>& m2);
	    
		template <typename T> 
		tmat4x4<T> operator- (const tmat4x4<T>& m, const T & s);

		template <typename T> 
		tmat4x4<T> operator- (const T & s, const tmat4x4<T>& m);

		template <typename T> 
		tmat4x4<T> operator- (const tmat4x4<T>& m1, const tmat4x4<T>& m2);

		template <typename T> 
		tmat4x4<T> operator* (const tmat4x4<T>& m, const T & s);

		template <typename T> 
		tmat4x4<T> operator* (const T & s, const tmat4x4<T>& m);

		template <typename T> 
		detail::tvec4<T> operator* (const tmat4x4<T>& m, const detail::tvec4<T>& v);

		template <typename T> 
		detail::tvec4<T> operator* (const detail::tvec4<T>& v, const tmat4x4<T>& m);

		template <typename T> 
		tmat4x4<T> operator* (const tmat4x4<T>& m1, const tmat4x4<T>& m2);

		template <typename T> 
		tmat4x4<T> operator/ (const tmat4x4<T>& m, const T & s);

		template <typename T> 
		tmat4x4<T> operator/ (const T & s, const tmat4x4<T>& m);

		template <typename T> 
		detail::tvec4<T> operator/ (const tmat4x4<T>& m, const detail::tvec4<T>& v);

		template <typename T> 
		detail::tvec4<T> operator/ (const detail::tvec4<T>& v, const tmat4x4<T>& m);

		template <typename T> 
		tmat4x4<T> operator/ (const tmat4x4<T>& m1, const tmat4x4<T>& m2);

		// Unary constant operators
		template <typename valType> 
		tmat4x4<valType> const operator-  (tmat4x4<valType> const & m);

		template <typename valType> 
		tmat4x4<valType> const operator-- (tmat4x4<valType> const & m, int);

		template <typename valType> 
		tmat4x4<valType> const operator++ (tmat4x4<valType> const & m, int);

	} //namespace detail

	namespace core{
	namespace type{
	namespace matrix{

	namespace precision
	{
		//! 4 columns of 4 components matrix of low precision floating-point numbers.
		//! There is no garanty on the actual precision.
		//! (From GLSL 1.30.8 specification, section 4.1.6 Matrices and section 4.5 Precision and Precision Qualifiers)
		typedef detail::tmat4x4<lowp_float>		lowp_mat4x4;
		//! 4 columns of 4 components matrix of medium precision floating-point numbers.
		//! There is no garanty on the actual precision.
		//! (From GLSL 1.30.8 specification, section 4.1.6 Matrices and section 4.5 Precision and Precision Qualifiers)
		typedef detail::tmat4x4<mediump_float>	mediump_mat4x4;
		//! 4 columns of 4 components matrix of high precision floating-point numbers.
		//! There is no garanty on the actual precision.
		//! (From GLSL 1.30.8 specification, section 4.1.6 Matrices and section 4.5 Precision and Precision Qualifiers)
		typedef detail::tmat4x4<highp_float>	highp_mat4x4;
	}
	//namespace precision

#ifndef GLM_PRECISION
	//! 4 columns of 4 components matrix of floating-point numbers. 
	//! (From GLSL 1.30.8 specification, section 4.1.6 Matrices)
	typedef detail::tmat4x4<mediump_float>	mat4x4;
#elif(GLM_PRECISION & GLM_PRECISION_HIGH)
	typedef detail::tmat4x4<highp_float>	mat4x4;
#elif(GLM_PRECISION & GLM_PRECISION_MEDIUM)
	typedef detail::tmat4x4<mediump_float>	mat4x4;
#elif(GLM_PRECISION & GLM_PRECISION_LOW)
	typedef detail::tmat4x4<lowp_float>		mat4x4;
#else
	typedef detail::tmat4x4<mediump_float>	mat4x4;
#endif//GLM_PRECISION

	//! 4 columns of 4 components matrix of floating-point numbers. 
	//! (From GLSL 1.30.8 specification, section 4.1.6 Matrices)
	typedef mat4x4							mat4;

	}//namespace matrix
	}//namespace type
	}//namespace core
} //namespace glm

#include "type_mat4x4.inl"

#endif //glm_core_type_mat4x4
