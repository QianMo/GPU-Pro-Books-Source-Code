///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2008 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2005-01-14
// Updated : 2009-05-01
// Licence : This source is under MIT License
// File    : glm/glm.hpp
///////////////////////////////////////////////////////////////////////////////////////////////////

/*! \mainpage OpenGL Mathematics
 *
 * OpenGL Mathematics (GLM) is a C++ mathematics library for 3D applications based on the OpenGL Shading Language (GLSL) specification. 
 *
 * The goal of the project is to provide to 3D programmers math classes and functions that miss in C++ when we use to program with GLSL or any high level GPU language. With GLM, the idea is to have a library that works the same way that GLSL which imply a strict following of GLSL specification for the implementation.
 *
 * However, this project isn't limited by GLSL features. An extension system based on GLSL extensions development conventions allows to extend GLSL capabilities.
 *
 * GLM is release under MIT license and available for all version of GCC from version 3.4 and Visual Studio from version 8.0 as a platform independent library.
 *
 * Any feedback is welcome, please send them to g.truc.creation[NO_SPAM_THANKS]gmail.com.
 *
 */

#ifndef glm_glm
#define glm_glm

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

#define GLMvalType typename genType::value_type
#define GLMcolType typename genType::col_type
#define GLMrowType typename genType::row_type

#define GLMsizeType typename genType::size_type
#define GLMrowSize typename genType::row_size
#define GLMcolSize typename genType::col_size

#include <cmath>
#include <climits>
#include <limits>
#include "./setup.hpp"

//! GLM namespace, it contains all GLSL based features.
namespace glm
{
	namespace test
	{
		bool main_bug();
		bool main_core();
	}//namespace test

	//! GLM core. Namespace that includes all the feature define by GLSL 1.30.8 specification. This namespace is included in glm namespace.
	namespace core
	{
		//! Scalar, vectors and matrices 
		//! from section 4.1.2 Booleans, 4.1.3 Integers section, 4.1.4 Floats section,
		//! 4.1.5 Vectors and section 4.1.6 Matrices of GLSL 1.30.8 specification. 
		//! This namespace resolves precision qualifier define in section 4.5 of GLSL 1.30.8 specification.
		namespace type
		{
			//! Scalar types from section 4.1.2 Booleans, 4.1.3 Integers and 4.1.4 Floats of GLSL 1.30.8 specification. 
			//! This namespace is included in glm namespace.
			namespace scalar
			{
				//! Scalar types with precision qualifier.
				//! This namespace is included in glm namespace.
				namespace precision{}
			}

			//! Vector types from section 4.1.5 of GLSL 1.30.8 specification. 
			//! This namespace is included in glm namespace.
			namespace vector
			{
				//! Vector types with precision qualifier.
				//! This namespace is included in glm namespace.
				namespace precision{}
			}
			
			//! Matrix types from section 4.1.6 of GLSL 1.30.8 specification. 
			//! This namespace is included in glm namespace.
			namespace matrix
			{
				//! Matrix types with precision qualifier.
				//! This namespace is included in glm namespace.
				namespace precision{}
			}
		}
		//! Some of the functions defined in section 8 Built-in Functions of GLSL 1.30.8 specification.
		//! Angle and trigonometry, exponential, common, geometric, matrix and vector relational functions.
		namespace function{}
	}
	//namespace core

	using namespace core::type::scalar;
	using namespace core::type::scalar::precision;
	using namespace core::type::vector;
	using namespace core::type::vector::precision;
	using namespace core::type::matrix;
	using namespace core::type::matrix::precision;

	//! GLM experimental extensions. The interface could change between releases.
	namespace gtx{}

	//! GLM stable extensions.
	namespace gtc{}

	//! IMG extensions.
	namespace img{}

	//! VIRTREV extensions.
	namespace img{}

} //namespace glm

#include "./core/_detail.hpp"
#include "./core/type.hpp"
#include "./core/type_half.hpp"

#include "./core/func_common.hpp"
#include "./core/func_exponential.hpp"
#include "./core/func_geometric.hpp"
#include "./core/func_matrix.hpp"
#include "./core/func_trigonometric.hpp"
#include "./core/func_vector_relational.hpp"
#include "./core/func_noise.hpp"
#include "./core/_swizzle.hpp"
//#include "./core/_xref2.hpp"
//#include "./core/_xref3.hpp"
//#include "./core/_xref4.hpp"

#if(defined(GLM_MESSAGE) && (GLM_MESSAGE & (GLM_MESSAGE_CORE | GLM_MESSAGE_NOTIFICATION)))
#	pragma message("GLM message: Core library included")
#endif//GLM_MESSAGE

#if(defined(GLM_COMPILER) && (GLM_COMPILER & GLM_COMPILER_VC))

#define GLM_DEPRECATED __declspec(deprecated)
#define GLM_RESTRICT __restrict
#define GLM_ALIGN(x) __declspec(align(x))

//#define aligned(x) __declspec(align(x)) struct

#else

#define GLM_DEPRECATED
#define GLM_RESTRICT
#define GLM_ALIGN(x)

#endif//GLM_COMPILER

////////////////////
// check type sizes
#ifndef GLM_STATIC_ASSERT_NULL
	GLM_STATIC_ASSERT(sizeof(glm::detail::int8)==1);
	GLM_STATIC_ASSERT(sizeof(glm::detail::int16)==2);
	GLM_STATIC_ASSERT(sizeof(glm::detail::int32)==4);
	GLM_STATIC_ASSERT(sizeof(glm::detail::int64)==8);

	GLM_STATIC_ASSERT(sizeof(glm::detail::uint8)==1);
	GLM_STATIC_ASSERT(sizeof(glm::detail::uint16)==2);
	GLM_STATIC_ASSERT(sizeof(glm::detail::uint32)==4);
	GLM_STATIC_ASSERT(sizeof(glm::detail::uint64)==8);

	GLM_STATIC_ASSERT(sizeof(glm::detail::float16)==2);
	GLM_STATIC_ASSERT(sizeof(glm::detail::float32)==4);
	GLM_STATIC_ASSERT(sizeof(glm::detail::float64)==8);
#endif//GLM_STATIC_ASSERT_NULL

#endif //glm_glm
