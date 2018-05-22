///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2009-05-07
// Updated : 2009-05-07
// Licence : This source is under MIT License
// File    : glm/gtx/simd_vec4.hpp
///////////////////////////////////////////////////////////////////////////////////////////////////
// Dependency:
// - GLM core
// - intrinsic
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_gtx_simd_mat4
#define glm_gtx_simd_mat4

// Dependency:
#include "../glm.hpp"
#include <xmmintrin.h>
#include <emmintrin.h>

namespace glm
{
	namespace detail
	{
		GLM_ALIGN(16) struct fmat4x4SIMD
		{
			static __m128 one;

			enum no_init
			{
				NO_INIT
			};

			typedef float value_type;
			typedef fvec4SIMD col_type;
			typedef fvec4SIMD row_type;
			typedef glm::sizeType size_type;
			static size_type value_size();
			static size_type col_size();
			static size_type row_size();
			static bool is_matrix();

			fvec4SIMD Data[4];

			//////////////////////////////////////
			// Constructors

			fmat4x4SIMD();
			explicit fmat4x4SIMD(float const & s);
			explicit fmat4x4SIMD(
				float const & x0, float const & y0, float const & z0, float const & w0,
				float const & x1, float const & y1, float const & z1, float const & w1,
				float const & x2, float const & y2, float const & z2, float const & w2,
				float const & x3, float const & y3, float const & z3, float const & w3);
			explicit fmat4x4SIMD(
				fvec4SIMD const & v0,
				fvec4SIMD const & v1,
				fvec4SIMD const & v2,
				fvec4SIMD const & v3);
			explicit fmat4x4SIMD(
				tmat4x4 const & m);

			// Conversions
			//template <typename U> 
			//explicit tmat4x4(tmat4x4<U> const & m);

			//explicit tmat4x4(tmat2x2<T> const & x);
			//explicit tmat4x4(tmat3x3<T> const & x);
			//explicit tmat4x4(tmat2x3<T> const & x);
			//explicit tmat4x4(tmat3x2<T> const & x);
			//explicit tmat4x4(tmat2x4<T> const & x);
			//explicit tmat4x4(tmat4x2<T> const & x);
			//explicit tmat4x4(tmat3x4<T> const & x);
			//explicit tmat4x4(tmat4x3<T> const & x);

			// Accesses
			fvec4SIMD & operator[](size_type i);
			fvec4SIMD const & operator[](size_type i) const;

			// Unary updatable operators
			fmat4x4SIMD & operator= (fmat4x4SIMD const & m);
			fmat4x4SIMD & operator+= (float const & s);
			fmat4x4SIMD & operator+= (fmat4x4SIMD const & m);
			fmat4x4SIMD & operator-= (float const & s);
			fmat4x4SIMD & operator-= (fmat4x4SIMD const & m);
			fmat4x4SIMD & operator*= (float const & s);
			fmat4x4SIMD & operator*= (fmat4x4SIMD const & m);
			fmat4x4SIMD & operator/= (float const & s);
			fmat4x4SIMD & operator/= (fmat4x4SIMD const & m);
			fmat4x4SIMD & operator++ ();
			fmat4x4SIMD & operator-- ();
		};

		// Binary operators
		fmat4x4SIMD operator+ (fmat4x4SIMD const & m, float const & s);
		fmat4x4SIMD operator+ (float const & s, fmat4x4SIMD const & m);
		fmat4x4SIMD operator+ (fmat4x4SIMD const & m1, fmat4x4SIMD const & m2);
	    
		fmat4x4SIMD operator- (fmat4x4SIMD const & m, float const & s);
		fmat4x4SIMD operator- (float const & s, fmat4x4SIMD const & m);
		fmat4x4SIMD operator- (fmat4x4SIMD const & m1, fmat4x4SIMD const & m2);

		fmat4x4SIMD operator* (fmat4x4SIMD const & m, float const & s);
		fmat4x4SIMD operator* (float const & s, fmat4x4SIMD const & m);

		fvec4SIMD operator* (fmat4x4SIMD const & m, fvec4SIMD const & v);
		fvec4SIMD operator* (fvec4SIMD const & v, fmat4x4SIMD const & m);

		fmat4x4SIMD operator* (fmat4x4SIMD const & m1, fmat4x4SIMD const & m2);

		fmat4x4SIMD operator/ (fmat4x4SIMD const & m, float const & s);
		fmat4x4SIMD operator/ (float const & s, fmat4x4SIMD const & m);

		fvec4SIMD operator/ (fmat4x4SIMD const & m, fvec4SIMD const & v);
		fvec4SIMD operator/ (fvec4SIMD const & v, fmat4x4SIMD const & m);

		fmat4x4SIMD operator/ (fmat4x4SIMD const & m1, fmat4x4SIMD const & m2);

		// Unary constant operators
		fmat4x4SIMD const operator-  (fmat4x4SIMD const & m);
		fmat4x4SIMD const operator-- (fmat4x4SIMD const & m, int);
		fmat4x4SIMD const operator++ (fmat4x4SIMD const & m, int);

	}//namespace detail

	namespace gtx{
	//! GLM_GTX_simd_mat4 extension: SIMD implementation of vec4 type.
	namespace simd_mat4
	{
		typedef detail::fmat4SIMD mat4SIMD;

	}//namespace simd_mat4
	}//namespace gtx
}//namespace glm

#define GLM_GTX_simd_mat4		namespace gtx::simd_mat4;
#ifndef GLM_GTX_GLOBAL
namespace glm {using GLM_GTX_simd_mat4;}
#endif//GLM_GTX_GLOBAL

#include "simd_mat4.inl"

#endif//glm_gtx_simd_mat4
