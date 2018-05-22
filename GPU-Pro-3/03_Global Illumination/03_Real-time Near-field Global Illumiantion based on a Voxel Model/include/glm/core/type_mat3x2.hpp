///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2006-08-05
// Updated : 2008-08-30
// Licence : This source is under MIT License
// File    : glm/core/type_mat3x2.h
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_core_type_mat3x2
#define glm_core_type_mat3x2

#include "type_size.hpp"

namespace glm
{
	namespace test
	{
		void main_mat3x2();
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

		//!< \brief Template for 3 * 2 matrix of floating-point numbers.
		template <typename T> 
		struct tmat3x2
		{
		public:
			typedef tmat3x2<T>* pointer;
			typedef const tmat3x2<T>* const_pointer;
			typedef const tmat3x2<T>*const const_pointer_const;
			typedef tmat3x2<T>*const pointer_const;
			typedef tmat3x2<T>& reference;
			typedef const tmat3x2<T>& const_reference;
			typedef const tmat3x2<T>& param_type;
			typedef tmat2x3<T> transpose_type;

			typedef T value_type;
			typedef detail::tvec3<T> col_type;
			typedef detail::tvec2<T> row_type;
			typedef glm::sizeType size_type;
			static size_type col_size();
			static size_type row_size();
			static bool is_matrix();

		public:
			tmat2x3<T> _inverse() const;

		private:
			// Data
			detail::tvec2<T> value[3];

		public:
			// Constructors
			tmat3x2();
			explicit tmat3x2(const T x);
			explicit tmat3x2(
				const T x0, const T y0,
				const T x1, const T y1,
				const T x2, const T y2);
			explicit tmat3x2(
				const detail::tvec2<T>& v0, 
				const detail::tvec2<T>& v1,
				const detail::tvec2<T>& v2);

			// Conversion
			template <typename U> 
			explicit tmat3x2(const tmat3x2<U>& m);

			explicit tmat3x2(const tmat2x2<T>& x);
			explicit tmat3x2(const tmat3x3<T>& x);
			explicit tmat3x2(const tmat4x4<T>& x);
			explicit tmat3x2(const tmat2x3<T>& x);
			explicit tmat3x2(const tmat2x4<T>& x);
			explicit tmat3x2(const tmat3x4<T>& x);
			explicit tmat3x2(const tmat4x2<T>& x);
			explicit tmat3x2(const tmat4x3<T>& x);

			// Accesses
			detail::tvec2<T>& operator[](size_type i);
			const detail::tvec2<T>& operator[](size_type i) const;

			// Unary updatable operators
			tmat3x2<T>& operator=  (const tmat3x2<T>& m);
			tmat3x2<T>& operator+= (const T & s);
			tmat3x2<T>& operator+= (const tmat3x2<T>& m);
			tmat3x2<T>& operator-= (const T & s);
			tmat3x2<T>& operator-= (const tmat3x2<T>& m);
			tmat3x2<T>& operator*= (const T & s);
			tmat3x2<T>& operator*= (const tmat2x3<T>& m);
			tmat3x2<T>& operator/= (const T & s);
			//tmat3x2<T>& operator/= (const tmat2x3<T>& m);

			tmat3x2<T>& operator++ ();
			tmat3x2<T>& operator-- ();
		};

		// Binary operators
		template <typename T> 
		tmat3x2<T> operator+ (const tmat3x2<T>& m, const T & s);
	    
		template <typename T> 
		tmat3x2<T> operator+ (const tmat3x2<T>& m1, const tmat3x2<T>& m2);
	    
		template <typename T> 
		tmat3x2<T> operator- (const tmat3x2<T>& m, const T & s);

		template <typename T> 
		tmat3x2<T> operator- (const tmat3x2<T>& m1, const tmat3x2<T>& m2);

		template <typename T> 
		tmat3x2<T> operator* (const tmat3x2<T>& m, const T & s);

		template <typename T> 
		tmat3x2<T> operator* (const T & s, const tmat3x4<T>& m);

		template <typename T>
		detail::tvec2<T> operator* (const tmat3x2<T>& m, const detail::tvec3<T>& v);

		template <typename T> 
		detail::tvec3<T> operator* (const detail::tvec2<T>& v, const tmat3x2<T>& m);

		template <typename T>
		tmat2x2<T> operator* (const tmat3x2<T>& m1, const tmat2x3<T>& m2);

		template <typename T> 
		tmat2x3<T> operator/ (const tmat2x3<T>& m, const T & s);

		template <typename T> 
		tmat2x3<T> operator/ (const T s, const tmat2x3<T>& m);

		//template <typename T> 
		//detail::tvec2<T> operator/ (const tmat3x2<T>& m, const detail::tvec3<T>& v);

		//template <typename T> 
		//detail::tvec3<T> operator/ (const detail::tvec2<T>& v, const tmat3x2<T>& m);

		//template <typename T> 
		//tmat2x2<T> operator/ (const tmat2x3<T>& m1, const tmat3x2<T>& m2);

		// Unary constant operators
		template <typename valType> 
		tmat3x2<valType> const operator-  (tmat3x2<valType> const & m);

		template <typename valType> 
		tmat3x2<valType> const operator-- (tmat3x2<valType> const & m, int);

		template <typename valType> 
		tmat3x2<valType> const operator++ (tmat3x2<valType> const & m, int);

	} //namespace detail

	namespace core{
	namespace type{
	namespace matrix{

	namespace precision
	{
		//! 3 columns of 2 components matrix of low precision floating-point numbers.
		//! There is no garanty on the actual precision.
		//! (From GLSL 1.30.8 specification, section 4.1.6 Matrices and section 4.5 Precision and Precision Qualifiers)
		typedef detail::tmat3x2<lowp_float>		lowp_mat3x2;
		//! 3 columns of 2 components matrix of medium precision floating-point numbers.
		//! There is no garanty on the actual precision.
		//! (From GLSL 1.30.8 specification, section 4.1.6 Matrices and section 4.5 Precision and Precision Qualifiers)
		typedef detail::tmat3x2<mediump_float>	mediump_mat3x2;
		//! 3 columns of 2 components matrix of high precision floating-point numbers.
		//! There is no garanty on the actual precision.
		//! (From GLSL 1.30.8 specification, section 4.1.6 Matrices and section 4.5 Precision and Precision Qualifiers)
		typedef detail::tmat3x2<highp_float>	highp_mat3x2;
	}
	//namespace precision

#ifndef GLM_PRECISION 
	//! 3 columns of 2 components matrix of floating-point numbers. 
	//! (From GLSL 1.30.8 specification, section 4.1.6 Matrices)
	typedef detail::tmat3x2<mediump_float>	mat3x2;
#elif(GLM_PRECISION & GLM_PRECISION_HIGH)
	typedef detail::tmat3x2<highp_float>	mat3x2;
#elif(GLM_PRECISION & GLM_PRECISION_MEDIUM)
	typedef detail::tmat3x2<mediump_float>	mat3x2;
#elif(GLM_PRECISION & GLM_PRECISION_LOW)
	typedef detail::tmat3x2<lowp_float>		mat3x2;
#else
	typedef detail::tmat3x2<mediump_float>	mat3x2;
#endif//GLM_PRECISION

	}//namespace matrix
	}//namespace type
	}//namespace core
} //namespace glm

#include "type_mat3x2.inl"

#endif //glm_core_type_mat3x2
