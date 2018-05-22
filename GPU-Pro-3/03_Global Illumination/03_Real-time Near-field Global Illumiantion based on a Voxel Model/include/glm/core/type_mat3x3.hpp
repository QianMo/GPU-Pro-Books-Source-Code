///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2005-01-27
// Updated : 2008-08-30
// Licence : This source is under MIT License
// File    : glm/core/type_mat3x3.h
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_core_type_mat3x3
#define glm_core_type_mat3x3

#include "type_size.hpp"

namespace glm
{
	namespace test
	{
		void main_mat3x3();
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

		//!< \brief Template for 3 * 3 matrix of floating-point numbers.
		template <typename T> 
		struct tmat3x3
		{
		public:
			typedef tmat3x3<T>* pointer;
			typedef const tmat3x3<T>* const_pointer;
			typedef const tmat3x3<T>*const const_pointer_const;
			typedef tmat3x3<T>*const pointer_const;
			typedef tmat3x3<T>& reference;
			typedef const tmat3x3<T>& const_reference;
			typedef const tmat3x3<T>& param_type;
			typedef tmat3x3<T> transpose_type;

			typedef T value_type;
			typedef detail::tvec3<T> col_type;
			typedef detail::tvec3<T> row_type;
			typedef glm::sizeType size_type;
			static size_type value_size();
			static size_type col_size();
			static size_type row_size();
			static bool is_matrix();

		public:
			tmat3x3<T> _inverse() const;

		private:
			// Data
			tvec3<T> value[3];

		public:
			// Constructors
			tmat3x3();
			explicit tmat3x3(const T x);
			explicit tmat3x3(
				const T x0, const T y0, const T z0,
				const T x1, const T y1, const T z1,
				const T x2, const T y2, const T z2);
			explicit tmat3x3(
				const detail::tvec3<T> & v0, 
				const detail::tvec3<T> & v1,
				const detail::tvec3<T> & v2);

			// Conversions
			template <typename U> 
			explicit tmat3x3(const tmat3x3<U>& m);

			explicit tmat3x3(const tmat2x2<T>& x);
			explicit tmat3x3(const tmat4x4<T>& x);
			explicit tmat3x3(const tmat2x3<T>& x);
			explicit tmat3x3(const tmat3x2<T>& x);
			explicit tmat3x3(const tmat2x4<T>& x);
			explicit tmat3x3(const tmat4x2<T>& x);
			explicit tmat3x3(const tmat3x4<T>& x);
			explicit tmat3x3(const tmat4x3<T>& x);

			// Accesses
			detail::tvec3<T>& operator[](size_type i);
			detail::tvec3<T> const & operator[](size_type i) const;

			// Unary updatable operators
			tmat3x3<T>& operator=(const tmat3x3<T>& m);
			tmat3x3<T>& operator+= (const T & s);
			tmat3x3<T>& operator+= (const tmat3x3<T>& m);
			tmat3x3<T>& operator-= (const T & s);
			tmat3x3<T>& operator-= (const tmat3x3<T>& m);
			tmat3x3<T>& operator*= (const T & s);
			tmat3x3<T>& operator*= (const tmat3x3<T>& m);
			tmat3x3<T>& operator/= (const T & s);
			tmat3x3<T>& operator/= (const tmat3x3<T>& m);
			tmat3x3<T>& operator++ ();
			tmat3x3<T>& operator-- ();
		};

		// Binary operators
		template <typename T> 
		tmat3x3<T> operator+ (const tmat3x3<T>& m, const T & s);

		template <typename T> 
		tmat3x3<T> operator+ (const T & s, const tmat3x3<T>& m);

		template <typename T> 
		tmat3x3<T> operator+ (const tmat3x3<T>& m1, const tmat3x3<T>& m2);
	    
		template <typename T> 
		tmat3x3<T> operator- (const tmat3x3<T>& m, const T & s);

		template <typename T> 
		tmat3x3<T> operator- (const T & s, const tmat3x3<T>& m);

		template <typename T> 
		tmat3x3<T> operator- (const tmat3x3<T>& m1, const tmat3x3<T>& m2);

		template <typename T> 
		tmat3x3<T> operator* (const tmat3x3<T>& m, const T & s);

		template <typename T> 
		tmat3x3<T> operator* (const T & s, const tmat3x3<T>& m);

		template <typename T> 
		detail::tvec3<T> operator* (const tmat3x3<T>& m, const detail::tvec3<T>& v);

		template <typename T> 
		detail::tvec3<T> operator* (const detail::tvec3<T>& v, const tmat3x3<T>& m);

		template <typename T> 
		tmat3x3<T> operator* (const tmat3x3<T>& m1, const tmat3x3<T>& m2);

		template <typename T> 
		tmat3x3<T> operator/ (const tmat3x3<T>& m, const T & s);

		template <typename T> 
		tmat3x3<T> operator/ (const T & s, const tmat3x3<T>& m);

		template <typename T> 
		detail::tvec3<T> operator/ (const tmat3x3<T>& m, const detail::tvec3<T>& v);

		template <typename T> 
		detail::tvec3<T> operator/ (const detail::tvec3<T>& v, const tmat3x3<T>& m);

		template <typename T> 
		tmat3x3<T> operator/ (const tmat3x3<T>& m1, const tmat3x3<T>& m2);

		// Unary constant operators
		template <typename valType> 
		tmat3x3<valType> const operator-  (tmat3x3<valType> const & m);

		template <typename valType> 
		tmat3x3<valType> const operator-- (tmat3x3<valType> const & m, int);

		template <typename valType> 
		tmat3x3<valType> const operator++ (tmat3x3<valType> const & m, int);

	} //namespace detail

	namespace core{
	namespace type{
	namespace matrix{

	namespace precision
	{
		//! 3 columns of 3 components matrix of low precision floating-point numbers.
		//! There is no garanty on the actual precision.
		//! (From GLSL 1.30.8 specification, section 4.1.6 Matrices and section 4.5 Precision and Precision Qualifiers)
		typedef detail::tmat3x3<lowp_float>		lowp_mat3x3;
		//! 3 columns of 3 components matrix of medium precision floating-point numbers.
		//! There is no garanty on the actual precision.
		//! (From GLSL 1.30.8 specification, section 4.1.6 Matrices and section 4.5 Precision and Precision Qualifiers)
		typedef detail::tmat3x3<mediump_float>	mediump_mat3x3;
		//! 3 columns of 3 components matrix of high precision floating-point numbers.
		//! There is no garanty on the actual precision.
		//! (From GLSL 1.30.8 specification, section 4.1.6 Matrices and section 4.5 Precision and Precision Qualifiers)
		typedef detail::tmat3x3<highp_float>	highp_mat3x3;
	}
	//namespace precision

#ifndef GLM_PRECISION
	//! 3 columns of 3 components matrix of floating-point numbers. 
	//! (From GLSL 1.30.8 specification, section 4.1.6 Matrices)
	typedef detail::tmat3x3<mediump_float>	mat3x3;
#elif(GLM_PRECISION & GLM_PRECISION_HIGH)
	typedef detail::tmat3x3<highp_float>	mat3x3;
#elif(GLM_PRECISION & GLM_PRECISION_MEDIUM)
	typedef detail::tmat3x3<mediump_float>	mat3x3;
#elif(GLM_PRECISION & GLM_PRECISION_LOW)
	typedef detail::tmat3x3<lowp_float>		mat3x3;
#else
	typedef detail::tmat3x3<mediump_float>	mat3x3;
#endif//GLM_PRECISION

	//! 3 columns of 3 components matrix of floating-point numbers. 
	//! (From GLSL 1.30.8 specification, section 4.1.6 Matrices)
	typedef mat3x3							mat3;

	}//namespace matrix
	}//namespace type
	}//namespace core
} //namespace glm

#include "type_mat3x3.inl"

#endif //glm_core_type_mat3x3
