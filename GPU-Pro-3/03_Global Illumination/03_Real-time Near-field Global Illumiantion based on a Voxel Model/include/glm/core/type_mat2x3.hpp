///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2006-10-01
// Updated : 2008-08-30
// Licence : This source is under MIT License
// File    : glm/core/type_mat2x3.h
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_core_type_mat2x3
#define glm_core_type_mat2x3

#include "type_size.hpp"

namespace glm
{
	namespace test
	{
		void main_mat2x3();
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

		//!< \brief Template for 2 * 3 matrix of floating-point numbers.
		template <typename T> 
		struct tmat2x3
		{
		public:
			typedef tmat2x3<T>* pointer;
			typedef const tmat2x3<T>* const_pointer;
			typedef const tmat2x3<T>*const const_pointer_const;
			typedef tmat2x3<T>*const pointer_const;
			typedef tmat2x3<T>& reference;
			typedef const tmat2x3<T>& const_reference;
			typedef const tmat2x3<T>& param_type;
			typedef tmat3x2<T> transpose_type;

			typedef T value_type;
			typedef detail::tvec3<T> col_type;
			typedef detail::tvec2<T> row_type;
			typedef glm::sizeType size_type;
			static size_type col_size();
			static size_type row_size();
			static bool is_matrix();

		public:
			tmat3x2<T> _inverse() const;

		private:
			// Data 
			detail::tvec3<T> value[2];

		public:
			// Constructors
			tmat2x3();
			explicit tmat2x3(const T x);
			explicit tmat2x3(
				const T x0, const T y0, const T z0,
				const T x1, const T y1, const T z1);
			explicit tmat2x3(
				const detail::tvec3<T>& v0, 
				const detail::tvec3<T>& v1);

			// Conversion
			template <typename U> 
			explicit tmat2x3(const tmat2x3<U>& m);

			explicit tmat2x3(const tmat2x2<T>& x);
			explicit tmat2x3(const tmat3x3<T>& x);
			explicit tmat2x3(const tmat4x4<T>& x);
			explicit tmat2x3(const tmat2x4<T>& x);
			explicit tmat2x3(const tmat3x2<T>& x);
			explicit tmat2x3(const tmat3x4<T>& x);
			explicit tmat2x3(const tmat4x2<T>& x);
			explicit tmat2x3(const tmat4x3<T>& x);

			// Accesses
			detail::tvec3<T>& operator[](size_type i);
			detail::tvec3<T> const & operator[](size_type i) const;

			// Unary updatable operators
			tmat2x3<T>& operator=  (const tmat2x3<T>& m);
			tmat2x3<T>& operator+= (const T & s);
			tmat2x3<T>& operator+= (const tmat2x3<T>& m);
			tmat2x3<T>& operator-= (const T & s);
			tmat2x3<T>& operator-= (const tmat2x3<T>& m);
			tmat2x3<T>& operator*= (const T & s);
			tmat2x3<T>& operator*= (const tmat3x2<T>& m);
			tmat2x3<T>& operator/= (const T & s);
			// tmat2x3<T>& operator/= (const tmat3x2<T>& m);

			tmat2x3<T>& operator++ ();
			tmat2x3<T>& operator-- ();

			// Unary constant operators
			const tmat2x3<T> operator- () const;
			const tmat2x3<T> operator++ (int n) const;
			const tmat2x3<T> operator-- (int n) const;
		};

		// Binary operators
		template <typename T> 
		tmat2x3<T> operator+ (const tmat2x3<T>& m, const T & s);
	    
		template <typename T> 
		tmat2x3<T> operator+ (const tmat2x3<T>& m1, const tmat2x3<T>& m2);
	    
		template <typename T> 
		tmat2x3<T> operator- (const tmat2x3<T>& m, const T & s);

		template <typename T> 
		tmat2x3<T> operator- (const tmat2x3<T>& m1, const tmat2x3<T>& m2);

		template <typename T> 
		tmat2x3<T> operator* (const tmat2x3<T>& m, const T & s);

		template <typename T> 
		tmat2x3<T> operator* (const T & s, const tmat2x3<T>& m);

		template <typename T>
		detail::tvec3<T> operator* (const tmat2x3<T>& m, const detail::tvec2<T>& v);

		template <typename T> 
		detail::tvec3<T> operator* (const detail::tvec3<T>& v, const tmat2x3<T>& m);

		template <typename T>
		tmat3x3<T> operator* (const tmat2x3<T>& m1, const tmat3x2<T>& m2);

		template <typename T> 
		tmat3x2<T> operator/ (const tmat2x3<T>& m, const T & s);

		template <typename T> 
		tmat3x2<T> operator/ (const T & s, const tmat2x3<T>& m);

		//template <typename T> 
		//detail::tvec3<T> operator/ (const tmat2x3<T>& m, const detail::tvec2<T>& v);

		//template <typename T> 
		//detail::tvec2<T> operator/ (const detail::tvec3<T>& v, const tmat2x3<T>& m);

		//template <typename T> 
		//tmat3x3<T> operator/ (const tmat3x2<T>& m1, const tmat2x3<T>& m2);

		// Unary constant operators
		template <typename valType> 
		tmat2x3<valType> const operator-  (tmat2x3<valType> const & m);

		template <typename valType> 
		tmat2x3<valType> const operator-- (tmat2x3<valType> const & m, int);

		template <typename valType> 
		tmat2x3<valType> const operator++ (tmat2x3<valType> const & m, int);

	} //namespace detail

	namespace core{
	namespace type{
	namespace matrix{

	namespace precision
	{
		//! 2 columns of 3 components matrix of low precision floating-point numbers.
		//! There is no garanty on the actual precision.
		//! (From GLSL 1.30.8 specification, section 4.1.6 Matrices and section 4.5 Precision and Precision Qualifiers)
		typedef detail::tmat2x3<lowp_float>		lowp_mat2x3;
		//! 2 columns of 3 components matrix of medium precision floating-point numbers. 
		//! There is no garanty on the actual precision.
		//! (From GLSL 1.30.8 specification, section 4.1.6 Matrices and section 4.5 Precision and Precision Qualifiers)
		typedef detail::tmat2x3<mediump_float>	mediump_mat2x3;
		//! 2 columns of 3 components matrix of high precision floating-point numbers.
		//! There is no garanty on the actual precision.
		//! (From GLSL 1.30.8 specification, section 4.1.6 Matrices and section 4.5 Precision and Precision Qualifiers)
		typedef detail::tmat2x3<highp_float>	highp_mat2x3;
	}
	//namespace precision

#ifndef GLM_PRECISION 
	//! 2 columns of 3 components matrix of floating-point numbers. 
	//! (From GLSL 1.30.8 specification, section 4.1.6 Matrices)
	typedef detail::tmat2x3<mediump_float>	mat2x3;
#elif(GLM_PRECISION & GLM_PRECISION_HIGHP_FLOAT)
	typedef detail::tmat2x3<highp_float>	mat2x3;
#elif(GLM_PRECISION & GLM_PRECISION_MEDIUMP_FLOAT)
	typedef detail::tmat2x3<mediump_float>	mat2x3;
#elif(GLM_PRECISION & GLM_PRECISION_LOWP_FLOAT)
	typedef detail::tmat2x3<lowp_float>		mat2x3;
#else
	typedef detail::tmat2x3<mediump_float>	mat2x3;
#endif//GLM_PRECISION

	}//namespace matrix
	}//namespace type
	}//namespace core
} //namespace glm

#include "type_mat2x3.inl"

#endif //glm_core_type_mat2x3
