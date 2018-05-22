#ifndef GLM_EXT_VIRTREV_GL_HPP
#define GLM_EXT_VIRTREV_GL_HPP

///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
// Virtrev SDK copyright matrem (matrem84.free.fr)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2008-04-24
// Updated : 2008-10-07
// Licence : This source is under MIT License
// File    : glm/ext/virtrev/gl.h
///////////////////////////////////////////////////////////////////////////////////////////////////
// Dependency:
// - GLM core
// - glew or glee or gl library header
///////////////////////////////////////////////////////////////////////////////////////////////////

#include "../glm.hpp"

#if !defined(GLM_DEPENDENCE) || !(GLM_DEPENDENCE & (GLM_DEPENDENCE_GLEW|GLM_DEPENDENCE_GLEE|GLM_DEPENDENCE_GL))
#error GLM_VIRTREV_gl requires OpenGL to build. GLM_DEPENDENCE doesn't define the dependence.
#endif//GLM_DEPENDENCE

namespace glm
{
	namespace virtrev_glmext
	{
	//! GLM_VIRTREV_gl extension: Vector & matrix integration with OpenGL.
	namespace gl
	{
		typedef detail::tvec2<GLfloat> gl_vec2; ///< vec2 for GLfloat OpenGL type
		typedef detail::tvec3<GLfloat> gl_vec3; ///< vec3 for GLfloat OpenGL type
		typedef detail::tvec4<GLfloat> gl_vec4; ///< vec4 for GLfloat OpenGL type

		typedef detail::tvec2<GLshort> gl_svec2; ///< vec2 for GLshort OpenGL type
		typedef detail::tvec3<GLshort> gl_svec3; ///< vec3 for GLshort OpenGL type
		typedef detail::tvec4<GLshort> gl_svec4; ///< vec4 for GLshort OpenGL type

		typedef detail::tvec2<GLint> gl_ivec2; ///< vec2 for GLint OpenGL type
		typedef detail::tvec3<GLint> gl_ivec3; ///< vec3 for GLint OpenGL type
		typedef detail::tvec4<GLint> gl_ivec4; ///< vec4 for GLint OpenGL type

		typedef detail::tmat2x2<GLfloat> gl_mat2; ///< mat2x2 for GLfloat OpenGL type
		typedef detail::tmat3x3<GLfloat> gl_mat3; ///< mat3x3 for GLfloat OpenGL type
		typedef detail::tmat4x4<GLfloat> gl_mat4; ///< mat4x4 for GLfloat OpenGL type

		typedef detail::tmat2x3<GLfloat> gl_mat2x3; ///< mat2x3 for GLfloat OpenGL type
		typedef detail::tmat3x2<GLfloat> gl_mat3x2; ///< mat3x2 for GLfloat OpenGL type
		typedef detail::tmat2x4<GLfloat> gl_mat2x4; ///< mat2x4 for GLfloat OpenGL type
		typedef detail::tmat4x2<GLfloat> gl_mat4x2; ///< mat4x2 for GLfloat OpenGL type
		typedef detail::tmat3x4<GLfloat> gl_mat3x4; ///< mat3x4 for GLfloat OpenGL type
		typedef detail::tmat4x3<GLfloat> gl_mat4x3; ///< mat4x3 for GLfloat OpenGL type

	}
	}
}

#define GLM_VIRTREV_gl namespace glm::virtrev_glmext::gl
#ifndef GLM_VIRTREV_GLOBAL
namespace glm {using GLM_VIRTREV_gl;}
#endif//GLM_VIRTREV_GLOBAL

#endif//GLM_EXT_VIRTREV_GL_HPP

