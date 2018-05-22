///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2005-01-27
// Updated : 2008-08-30
// Licence : This source is under MIT License
// File    : glm/core/type_mat2x2.h
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_core_type_mat2x2
#define glm_core_type_mat2x2

#include "type_size.hpp"

namespace glm
{
	namespace test
	{
		void main_mat2x2();
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

		//!< \brief Template for 2 * 2 matrix of floating-point numbers.
		template <typename T> 
		struct tmat2x2
		{
		public:
			typedef tmat2x2<T>* pointer;
			typedef const tmat2x2<T>* const_pointer;
			typedef const tmat2x2<T>*const const_pointer_const;
			typedef tmat2x2<T>*const pointer_const;
			typedef tmat2x2<T>& reference;
			typedef const tmat2x2<T>& const_reference;
			typedef const tmat2x2<T>& param_type;
			typedef tmat2x2<T> transpose_type;

			typedef T value_type;
			typedef detail::tvec2<T> col_type;
			typedef detail::tvec2<T> row_type;
			typedef glm::sizeType size_type;
			static size_type value_size();
			static size_type col_size();
			static size_type row_size();
			static bool is_matrix();

		public:
			tmat2x2<T> _inverse() const;

		private:
			// Data 
			detail::tvec2<T> value[2];

		public:
			// Constructors
			tmat2x2();
			tmat2x2(tmat2x2<T> const & m);

			explicit tmat2x2(const T x);
			explicit tmat2x2(
				const T x1, const T y1, 
				const T x2, const T y2);
			explicit tmat2x2(
				const detail::tvec2<T> & v1, 
				const detail::tvec2<T> & v2);

			// Conversions
			template <typename U> 
			explicit tmat2x2(const tmat2x2<U>& m);

			explicit tmat2x2(const tmat3x3<T>& x);
			explicit tmat2x2(const tmat4x4<T>& x);
			explicit tmat2x2(const tmat2x3<T>& x);
			explicit tmat2x2(const tmat3x2<T>& x);
			explicit tmat2x2(const tmat2x4<T>& x);
			explicit tmat2x2(const tmat4x2<T>& x);
			explicit tmat2x2(const tmat3x4<T>& x);
			explicit tmat2x2(const tmat4x3<T>& x);

			//////////////////////////////////////
			// Accesses

			detail::tvec2<T>& operator[](size_type i);
			detail::tvec2<T> const & operator[](size_type i) const;

			// Unary updatable operators
			tmat2x2<T>& operator=(tmat2x2<T> const & m);
			tmat2x2<T>& operator+=(const T & s);
			tmat2x2<T>& operator+=(tmat2x2<T> const & m);
			tmat2x2<T>& operator-=(const T & s);
			tmat2x2<T>& operator-=(tmat2x2<T> const & m);
			tmat2x2<T>& operator*=(const T & s);
			tmat2x2<T>& operator*= (tmat2x2<T> const & m);
			tmat2x2<T>& operator/= (const T & s);
			tmat2x2<T>& operator/= (tmat2x2<T> const & m);
			tmat2x2<T>& operator++ ();
			tmat2x2<T>& operator-- ();
		};

		// Binary operators
		template <typename T> 
		tmat2x2<T> operator+ (tmat2x2<T> const & m, const T & s);

		template <typename T> 
		tmat2x2<T> operator+ (const T & s, tmat2x2<T> const & m);

		template <typename T> 
		tmat2x2<T> operator+ (tmat2x2<T> const & m1, tmat2x2<T> const & m2);
	    
		template <typename T> 
		tmat2x2<T> operator- (tmat2x2<T> const & m, const T & s);

		template <typename T> 
		tmat2x2<T> operator- (const T & s, tmat2x2<T> const & m);

		template <typename T> 
		tmat2x2<T> operator- (tmat2x2<T> const & m1, tmat2x2<T> const & m2);

		template <typename T> 
		tmat2x2<T> operator* (tmat2x2<T> const & m, const T & s);

		template <typename T> 
		tmat2x2<T> operator* (const T & s, tmat2x2<T> const & m);

		template <typename T> 
		tvec2<T> operator* (tmat2x2<T> const & m, const tvec2<T>& v);

		template <typename T> 
		tvec2<T> operator* (const tvec2<T>& v, tmat2x2<T> const & m);

		template <typename T> 
		tmat2x2<T> operator* (tmat2x2<T> const & m1, tmat2x2<T> const & m2);

		template <typename T> 
		tmat2x2<T> operator/ (tmat2x2<T> const & m, const T & s);

		template <typename T> 
		tmat2x2<T> operator/ (const T & s, tmat2x2<T> const & m);

		template <typename T> 
		tvec2<T> operator/ (tmat2x2<T> const & m, const tvec2<T>& v);

		template <typename T> 
		tvec2<T> operator/ (const tvec2<T>& v, tmat2x2<T> const & m);

		template <typename T> 
		tmat2x2<T> operator/ (tmat2x2<T> const & m1, tmat2x2<T> const & m2);

		// Unary constant operators
		template <typename valType> 
		tmat2x2<valType> const operator-  (tmat2x2<valType> const & m);

		template <typename valType> 
		tmat2x2<valType> const operator-- (tmat2x2<valType> const & m, int);

		template <typename valType> 
		tmat2x2<valType> const operator++ (tmat2x2<valType> const & m, int);

	} //namespace detail

	namespace core{
	namespace type{
	namespace matrix{

	namespace precision
	{
		//! 2 columns of 2 components matrix of low precision floating-point numbers.
		//! There is no garanty on the actual precision.
		//! (From GLSL 1.30.8 specification, section 4.1.6 Matrices and section 4.5 Precision and Precision Qualifiers)
		typedef detail::tmat2x2<lowp_float>		lowp_mat2x2;
		//! 2 columns of 2 components matrix of medium precision floating-point numbers. 
		//! There is no garanty on the actual precision.
		//! (From GLSL 1.30.8 specification, section 4.1.6 Matrices and section 4.5 Precision and Precision Qualifiers)
		typedef detail::tmat2x2<mediump_float>	mediump_mat2x2;
		//! 2 columns of 2 components matrix of high precision floating-point numbers. 
		//! There is no garanty on the actual precision. 
		//! (From GLSL 1.30.8 specification, section 4.1.6 Matrices and section 4.5 Precision and Precision Qualifiers)
		typedef detail::tmat2x2<highp_float>	highp_mat2x2;
	}
	//namespace precision

#ifndef GLM_PRECISION
	//! 2 columns of 2 components matrix of floating-point numbers. 
	//! (From GLSL 1.30.8 specification, section 4.1.6 Matrices)
	typedef detail::tmat2x2<mediump_float>	mat2x2;
#elif(GLM_PRECISION & GLM_PRECISION_HIGH)
	typedef detail::tmat2x2<highp_float>	mat2x2;
#elif(GLM_PRECISION & GLM_PRECISION_MEDIUM)
	typedef detail::tmat2x2<mediump_float>	mat2x2;
#elif(GLM_PRECISION & GLM_PRECISION_LOW)
	typedef detail::tmat2x2<lowp_float>		mat2x2;
#else
	typedef detail::tmat2x2<mediump_float>	mat2x2;
#endif//GLM_PRECISION

	//! 2 columns of 2 components matrix of floating-point numbers. 
	//! (From GLSL 1.30.8 specification, section 4.1.6 Matrices)
	typedef mat2x2							mat2;

	}//namespace matrix
	}//namespace type
	}//namespace core
} //namespace glm

#include "type_mat2x2.inl"

#endif //glm_core_type_mat2x2
