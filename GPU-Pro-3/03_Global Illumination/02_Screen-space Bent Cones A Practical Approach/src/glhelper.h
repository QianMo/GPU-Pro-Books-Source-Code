#pragma once

//////////////////////////////////////////////////////////////////////////
// loading of opengl 3.3 extensions

// OpenGL
#ifdef WIN32
//#	define GLEW_STATIC
#	include <GL/glew.h>
#	include <GL/wglew.h>
//#	include <GL/glext.h>
//#	define glhGetProcAddress wglGetProcAddress
#elif defined(linux) || defined(__linux)
#	define GL_GLEXT_PROTOTYPES 1
#	include <GL/gl.h>
#	include <GL/glext.h>
#endif

// STL
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>

#include "config.h"

inline bool checkFramebufferStatus(std::string& result) {
    const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    switch(status) {
    case GL_FRAMEBUFFER_COMPLETE:
        result = "framebuffer ready"; // Status:  GL_FRAMEBUFFER_COMPLETE ... continuing, because everything is fine ...
        return true;
    case GL_FRAMEBUFFER_UNSUPPORTED:
        result = "GL_FRAMEBUFFER_UNSUPPORTED (i.e., this combination of formats doesn't work)";
        return false;
    case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
        result = "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT";
        return false;
    case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
        result = "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT";
        return false;
    case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
        result = "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER";
        return false;
    case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
        result = "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER";
        return false;
    case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
        result = "GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS";
        return false;
    case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
        result = "GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE";
        return false;
    default:
        result = "frame buffer object status invalid";
        return false;
    }
}

GLenum internalFormatToFormat(GLint internalFormat);
GLenum internalFormatToType(GLint internalFormat);
bool checkFramebufferStatus(unsigned int lineNbr, const std::string& file, const std::string& func);
bool checkError(const char* Title);
bool checkProgram(GLuint ProgramName, const std::string& output = std::string());
bool checkShader(GLuint ShaderName, const std::string& file);
bool validateProgram(GLuint ProgramName);

std::string floatToString(float value);
GLuint createShaderFromSource(GLenum type, const std::string& source, const std::string& output = std::string());
GLuint createShaderFromSourceWithMacros(GLenum type, const std::string& source, const std::vector<std::pair<std::string, std::string>>& macroReplacements, const std::string& output = std::string());
GLuint createShaderFromFile(GLenum type, const std::string& file);
GLuint createShaderWithDefines(GLenum type, const std::string& file, const std::vector<std::string>& defines);
GLuint createShaderWithLib(GLenum type, const std::string& shaderFile, const std::string& libFile, const std::vector<std::string>& defines = std::vector<std::string>());
GLuint createShaderWithMacro(GLenum type, const std::string& shaderFile, std::pair<std::string, std::string>& macroReplacement, const std::vector<std::string>& defines = std::vector<std::string>());
GLuint createShaderWithLibMacros(GLenum type, const std::string& shaderFile, const std::string& libFile, const std::vector<std::pair<std::string, std::string>>& macroReplacements, const std::vector<std::string>& defines = std::vector<std::string>());
GLuint createShaderWithMacros(GLenum type, const std::string& shaderFile, const std::vector<std::pair<std::string, std::string>>& macroReplacements, const std::vector<std::string>& defines = std::vector<std::string>());
GLuint createProgram(const GLuint vertShader, const GLuint fragShader, const std::string& output = std::string());

GLuint createProgram(
    std::string const & vertShaderFile, 
    std::string const & fragShaderFile);

void storeBoundTexturesSamplers(short count, std::vector<GLint>& outCurrentSamplers, std::vector<GLint>& outCurrentTextures, std::vector<GLint>& outOffsets);
void storeBoundTexturesSamplersAtOffset(short count, unsigned int offset, std::vector<GLint>& outCurrentSamplers, std::vector<GLint>& outCurrentTextures, std::vector<GLint>& outOffsets);
void storeBoundTexturesSamplers(short count, std::vector<GLint>& outCurrentSamplers, std::vector<GLint>& outCurrentTextures, std::vector<GLint>& outOffsets, GLint locked1);
void storeBoundTexturesSamplers(short count, std::vector<GLint>& outCurrentSamplers, std::vector<GLint>& outCurrentTextures, std::vector<GLint>& outOffsets, GLint locked1, GLint locked2);
void storeBoundTexturesSamplers(short count, std::vector<GLint>& outCurrentSamplers, std::vector<GLint>& outCurrentTextures, std::vector<GLint>& outOffsets, GLint locked1, GLint locked2, GLint locked3);
void storeBoundTexturesSamplers(short count, std::vector<GLint>& outCurrentSamplers, std::vector<GLint>& outCurrentTextures, std::vector<GLint>& outOffsets, const std::vector<GLint>& locked);
void restoreBoundTexturesSamplers(const std::vector<GLint>& currentSamplers, const std::vector<GLint>& currentTextures, const std::vector<GLint>& offsets);

#ifdef WIN32
#ifndef APIENTRY
#define APIENTRY
#endif
#ifndef APIENTRYP
#define APIENTRYP APIENTRY *
#endif
#ifndef GLAPI
#define GLAPI extern
#endif

typedef void (APIENTRYP PFNGLGETSAMPLERPARAMETERIFVPROC) (GLuint sampler, GLenum pname, GLfloat *params);
extern PFNGLGETSAMPLERPARAMETERIFVPROC glGetSamplerParameterIfv;

#endif//WIN32

