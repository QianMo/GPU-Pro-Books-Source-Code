///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2008 G-Truc Creation (www.g-truc.net)
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2006-11-13
// Updated : 2009-08-24
// Licence : This source is under MIT License
// File    : glm/setup.hpp
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_setup
#define glm_setup

///////////////////////////////////////////////////////////////////////////////////////////////////
// Version

#define GLM_VERSION					84
#define GLM_REVISION				1010

///////////////////////////////////////////////////////////////////////////////////////////////////
// Common values

#define GLM_DISABLE					0x00000000
#define GLM_ENABLE					0x00000001

///////////////////////////////////////////////////////////////////////////////////////////////////
// Message

#define GLM_MESSAGE_QUIET			0x00000000

#define GLM_MESSAGE_WARNING			0x00000001
#define GLM_MESSAGE_NOTIFICATION	0x00000002
#define GLM_MESSAGE_CORE			0x00000004
#define GLM_MESSAGE_EXTS			0x00000008
#define GLM_MESSAGE_SETUP			0x00000010

#define GLM_MESSAGE_ALL				GLM_MESSAGE_WARNING | GLM_MESSAGE_NOTIFICATION | GLM_MESSAGE_CORE | GLM_MESSAGE_EXTS | GLM_MESSAGE_SETUP

//! By default:
// #define GLM_MESSAGE GLM_MESSAGE_QUIET

///////////////////////////////////////////////////////////////////////////////////////////////////
// Precision

#define GLM_PRECISION_NONE			0x00000000

#define GLM_PRECISION_LOWP_FLOAT	0x00000011
#define GLM_PRECISION_MEDIUMP_FLOAT	0x00000012
#define GLM_PRECISION_HIGHP_FLOAT	0x00000013	

#define GLM_PRECISION_LOWP_INT		0x00001100
#define GLM_PRECISION_MEDIUMP_INT	0x00001200
#define GLM_PRECISION_HIGHP_INT		0x00001300

#define GLM_PRECISION_LOWP_UINT		0x00110000
#define GLM_PRECISION_MEDIUMP_UINT	0x00120000
#define GLM_PRECISION_HIGHP_UINT	0x00130000	

///////////////////////////////////////////////////////////////////////////////////////////////////
// Use options

// To disable multiple vector component names access.
// GLM_USE_ONLY_XYZW

// To use anonymous union to provide multiple component names access for class valType. Visual C++ only.
// GLM_USE_ANONYMOUS_UNION

///////////////////////////////////////////////////////////////////////////////////////////////////
// Compiler

#define GLM_COMPILER_NONE			0x00000000

// Visual C++ defines
#define GLM_COMPILER_VC				0x01000000
#define GLM_COMPILER_VC60			0x01000040 // unsupported
#define GLM_COMPILER_VC70			0x01000080 // unsupported
#define GLM_COMPILER_VC71			0x01000100 // unsupported
#define GLM_COMPILER_VC80			0x01000200
#define GLM_COMPILER_VC90			0x01000400
#define GLM_COMPILER_VC2010			0x01000800

// GCC defines
#define GLM_COMPILER_GCC            0x02000000
#define GLM_COMPILER_GCC28			0x02000020 // unsupported
#define GLM_COMPILER_GCC29			0x02000040 // unsupported
#define GLM_COMPILER_GCC30			0x02000080 // unsupported
#define GLM_COMPILER_GCC31			0x02000100 // unsupported
#define GLM_COMPILER_GCC32			0x02000200
#define GLM_COMPILER_GCC33			0x02000400
#define GLM_COMPILER_GCC34			0x02000800
#define GLM_COMPILER_GCC35			0x02001000
#define GLM_COMPILER_GCC40			0x02002000
#define GLM_COMPILER_GCC41			0x02004000
#define GLM_COMPILER_GCC42			0x02008000
#define GLM_COMPILER_GCC43			0x02010000
#define GLM_COMPILER_GCC44			0x02020000
#define GLM_COMPILER_GCC45			0x02040000
#define GLM_COMPILER_GCC46			0x02080000
#define GLM_COMPILER_GCC50			0x0210000

#define GLM_MODEL_32				0x00000010
#define GLM_MODEL_64				0x00000020

#ifndef GLM_COMPILER

/////////////////
// Visual C++ //

#ifdef _MSC_VER

#if defined(_WIN64)
#define GLM_MODEL	GLM_MODEL_64
#else
#define GLM_MODEL	GLM_MODEL_32
#endif//

#if _MSC_VER == 1200
#define GLM_COMPILER GLM_COMPILER_VC60
#endif

#if _MSC_VER == 1300
#define GLM_COMPILER GLM_COMPILER_VC70
#endif

#if _MSC_VER == 1310
#define GLM_COMPILER GLM_COMPILER_VC71
#endif

#if _MSC_VER == 1400
#define GLM_COMPILER GLM_COMPILER_VC80
#endif

#if _MSC_VER == 1500
#define GLM_COMPILER GLM_COMPILER_VC90
#endif

#if _MSC_VER == 1600
#define GLM_COMPILER GLM_COMPILER_VC2010
#endif

#endif//_MSC_VER

//////////////////
// GCC defines //

#ifdef __GNUC__

#if(defined(__WORDSIZE) && (__WORDSIZE == 64)) || defined(__arch64__)
#define GLM_MODEL	GLM_MODEL_64
#else
#define GLM_MODEL	GLM_MODEL_32
#endif//

#if (__GNUC__ == 2) && (__GNUC_MINOR__ == 8)
#error "GCC 2.8x isn't supported"
#define GLM_COMPILER GLM_COMPILER_GCC28
#endif

#if (__GNUC__ == 2) && (__GNUC_MINOR__ == 9)
#error "GCC 2.9x isn't supported"
#define GLM_COMPILER GLM_COMPILER_GCC29
#endif

#if (__GNUC__ == 3) && (__GNUC_MINOR__ == 0)
#error "GCC 3.0 isn't supported"
#define GLM_COMPILER GLM_COMPILER_GCC30
#endif

#if (__GNUC__ == 3) && (__GNUC_MINOR__ == 1)
#error "GCC 3.1 isn't supported"
#define GLM_COMPILER GLM_COMPILER_GCC31
#endif

#if (__GNUC__ == 3) && (__GNUC_MINOR__ == 2)
#define GLM_COMPILER GLM_COMPILER_GCC32
#endif

#if (__GNUC__ == 3) && (__GNUC_MINOR__ == 3)
#define GLM_COMPILER GLM_COMPILER_GCC33
#endif

#if (__GNUC__ == 3) && (__GNUC_MINOR__ == 4)
#define GLM_COMPILER GLM_COMPILER_GCC34
#endif

#if (__GNUC__ == 3) && (__GNUC_MINOR__ == 5)
#define GLM_COMPILER GLM_COMPILER_GCC35
#endif

#if (__GNUC__ == 4) && (__GNUC_MINOR__ == 0)
#define GLM_COMPILER GLM_COMPILER_GCC40
#endif

#if (__GNUC__ == 4) && (__GNUC_MINOR__ == 1)
#define GLM_COMPILER GLM_COMPILER_GCC41
#endif

#if (__GNUC__ == 4) && (__GNUC_MINOR__ == 2)
#define GLM_COMPILER GLM_COMPILER_GCC42
#endif

#if (__GNUC__ == 4) && (__GNUC_MINOR__ == 3)
#define GLM_COMPILER GLM_COMPILER_GCC43
#endif

#if (__GNUC__ == 4) && (__GNUC_MINOR__ == 4)
#define GLM_COMPILER GLM_COMPILER_GCC44
#endif

#if (__GNUC__ == 4) && (__GNUC_MINOR__ == 5)
#define GLM_COMPILER GLM_COMPILER_GCC45
#endif

#if (__GNUC__ == 4) && (__GNUC_MINOR__ == 6)
#define GLM_COMPILER GLM_COMPILER_GCC46
#endif

#if (__GNUC__ == 5) && (__GNUC_MINOR__ == 0)
#define GLM_COMPILER GLM_COMPILER_GCC50
#endif

#endif//__GNUC__

#endif//GLM_COMPILER

#ifndef GLM_COMPILER
#error "GLM_COMPILER undefined, your compiler may not be supported by GLM. Add #define GLM_COMPILER 0 to ignore this message."
#endif//GLM_COMPILER

#if(!defined(GLM_MODEL) && GLM_COMPILER != 0)
#error "GLM_MODEL undefined, your compiler may not be supported by GLM. Add #define GLM_MODEL 0 to ignore this message."
#endif//GLM_MODEL

#if(defined(GLM_MESSAGE) && (GLM_MESSAGE & (GLM_MESSAGE_SETUP | GLM_MESSAGE_NOTIFICATION)))
#	if(defined(GLM_COMPILER) && GLM_COMPILER & GLM_COMPILER_VC)
#		pragma message("GLM message: Compiled with Visual C++")
#	elif(defined(GLM_COMPILER) && GLM_COMPILER & GLM_COMPILER_GCC)
#		pragma message("GLM message: Compiled with GCC")
#	else
#		pragma message("GLM warning: Compiler not detected")
#	endif
#endif//GLM_MESSAGE

#if(defined(GLM_MESSAGE) && (GLM_MESSAGE & (GLM_MESSAGE_SETUP | GLM_MESSAGE_NOTIFICATION)))
#	if(GLM_MODEL == GLM_MODEL_64)
#		pragma message("GLM message: 64 bits model")
#	elif(GLM_MODEL == GLM_MODEL_32)
#		pragma message("GLM message: 32 bits model")
#	endif//GLM_MODEL
#endif//GLM_MESSAGE

///////////////////////////////////////////////////////////////////////////////////////////////////
// Compatibility

#define GLM_COMPATIBILITY_DEFAULT	0
#define GLM_COMPATIBILITY_STRICT	1

//! By default:
//#define GLM_COMPATIBILITY			GLM_COMPATIBILITY_DEFAULT

#if(defined(GLM_MESSAGE) && (GLM_MESSAGE & (GLM_MESSAGE_SETUP | GLM_MESSAGE_NOTIFICATION)))
#	if(!defined(GLM_COMPATIBILITY) || (defined(GLM_COMPATIBILITY) && (GLM_COMPATIBILITY == GLM_COMPATIBILITY_STRICT)))
#
#	elif(defined(GLM_COMPATIBILITY) && (GLM_COMPATIBILITY == GLM_COMPATIBILITY_STRICT))
#		pragma message("GLM message: compatibility strict")
#	endif//GLM_AUTO_CAST
#endif//GLM_MESSAGE

///////////////////////////////////////////////////////////////////////////////////////////////////
// External dependencies

#define GLM_DEPENDENCE_NONE			0x00000000
#define GLM_DEPENDENCE_GLEW			0x00000001
#define GLM_DEPENDENCE_GLEE			0x00000002
#define GLM_DEPENDENCE_GL			0x00000004
#define GLM_DEPENDENCE_GL3			0x00000008
#define GLM_DEPENDENCE_BOOST		0x00000010
#define GLM_DEPENDENCE_STL			0x00000020
#define GLM_DEPENDENCE_TR1			0x00000040
#define GLM_DEPENDENCE_TR2			0x00000080

//! By default:
// #define GLM_DEPENDENCE GLM_DEPENDENCE_NONE

#if(defined(GLM_DEPENDENCE) && (GLM_DEPENDENCE & GLM_DEPENDENCE_GLEW))
#include <GL/glew.hpp>
#elif(defined(GLM_DEPENDENCE) && (GLM_DEPENDENCE & GLM_DEPENDENCE_GLEE))
#include <GL/GLee.hpp> 
#elif(defined(GLM_DEPENDENCE) && (GLM_DEPENDENCE & GLM_DEPENDENCE_GL))
#include <GL/gl.h> 
#elif(defined(GLM_DEPENDENCE) && (GLM_DEPENDENCE & GLM_DEPENDENCE_GL3))
#include <GL3/gl3.h> 
#endif//GLM_DEPENDENCE

#if(defined(GLM_DEPENDENCE) && (GLM_DEPENDENCE & GLM_DEPENDENCE_BOOST))
#include <boost/static_assert.hpp>
#endif//GLM_DEPENDENCE

#if(defined(GLM_DEPENDENCE) && (GLM_DEPENDENCE & GLM_DEPENDENCE_BOOST)) || defined(BOOST_STATIC_ASSERT)
#define GLM_STATIC_ASSERT(x) BOOST_STATIC_ASSERT(x)
#else
#define GLM_STATIC_ASSERT(x) typedef char __CASSERT__##__LINE__[(x) ? 1 : -1]
#endif//GLM_DEPENDENCE

///////////////////////////////////////////////////////////////////////////////////////////////////
// Cast

#define GLM_CAST_NONE				0x00000000
#define GLM_CAST_DIRECTX_9			0x00000001
#define GLM_CAST_DIRECTX_10			0x00000002
#define GLM_CAST_NVSG				0x00000004
#define GLM_CAST_WILD_MAGIC_3		0x00000008
#define GLM_CAST_WILD_MAGIC_4		0x00000010
#define GLM_CAST_PHYSX				0x00000020
#define GLM_CAST_ODE				0x00000040

//! By default:
// #define GLM_CAST	GLM_CAST_NONE
// #define GLM_CAST_EXT	GLM_CAST_NONE

///////////////////////////////////////////////////////////////////////////////////////////////////
// Automatic cast
// glColor4fv(glm::vec4(1.0))

//! By default:
// #define GLM_AUTO_CAST			GLM_ENABLE

// GLM_AUTO_CAST isn't defined by defaut but also enable by default with GLM 0.7.x
// Disable GLM_AUTO_CAST by default on Visual C++ 7.1
#if(defined(GLM_COMPILER) && GLM_COMPILER & GLM_COMPILER_VC && GLM_COMPILER <= GLM_COMPILER_VC71)
#	if(defined(GLM_AUTO_CAST) || (GLM_AUTO_CAST == GLM_ENABLE))
#		error "GLM_AUTO_CAST isn't supported by Visual C++ 7.1 and below"
#	else
#		define GLM_AUTO_CAST GLM_DISABLE
#	endif//GLM_AUTO_CAST
#endif//GLM_COMPILER

#if(defined(GLM_MESSAGE) && (GLM_MESSAGE & (GLM_MESSAGE_SETUP | GLM_MESSAGE_NOTIFICATION)))
#	if(!defined(GLM_AUTO_CAST) || (defined(GLM_AUTO_CAST) && (GLM_AUTO_CAST == GLM_ENABLE)))
#		pragma message("GLM message: Auto cast enabled")
#	else
#		pragma message("GLM message: Auto cast disabled")
#	endif//GLM_AUTO_CAST
#endif//GLM_MESSAGE

///////////////////////////////////////////////////////////////////////////////////////////////////
// Swizzle operators

#define GLM_SWIZZLE_NONE            0x00000000
#define GLM_SWIZZLE_XYZW            0x00000002
#define GLM_SWIZZLE_RGBA            0x00000004
#define GLM_SWIZZLE_STQP            0x00000008
#define GLM_SWIZZLE_FULL            (GLM_SWIZZLE_XYZW | GLM_SWIZZLE_RGBA | GLM_SWIZZLE_STQP)

//! By default:
// #define GLM_SWIZZLE GLM_SWIZZLE_NONE

#if(defined(GLM_MESSAGE) && (GLM_MESSAGE & (GLM_MESSAGE_SETUP | GLM_MESSAGE_NOTIFICATION)))
#	if !defined(GLM_SWIZZLE)|| (defined(GLM_SWIZZLE) && GLM_SWIZZLE == GLM_SWIZZLE_NONE)
#		pragma message("GLM message: No swizzling operator used")
#	elif(defined(GLM_SWIZZLE) && GLM_SWIZZLE == GLM_SWIZZLE_FULL)
#		pragma message("GLM message: Full swizzling operator support enabled")
#	elif(defined(GLM_SWIZZLE) && GLM_SWIZZLE & GLM_SWIZZLE_FULL)
#		pragma message("GLM message: Partial swizzling operator support enabled")
#	endif//GLM_SWIZZLE
#endif//GLM_MESSAGE

///////////////////////////////////////////////////////////////////////////////////////////////////

#endif//glm_setup
