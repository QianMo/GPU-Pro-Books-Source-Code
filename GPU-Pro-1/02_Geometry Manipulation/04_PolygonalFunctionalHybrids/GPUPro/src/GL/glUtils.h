#ifndef _GL_UTILS_H_
#define _GL_UTILS_H_

#include	<GL/glew.h>

#include    <GL/gl.h>
#include    <GL/glu.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include    <string>

// a few quick macros to simplify the work with some shader constants
#define  GET_LOCATION(PROGRAM, NAME)                                             \
            static int location##NAME = glGetUniformLocationARB(PROGRAM, NAME);  \

#define  CALL_ARB_FUNCTION(PROGRAM, NAME, FUNCTION_ARB)                   \
            static int location##NAME = glGetUniformLocationARB(PROGRAM, #NAME); \
            if (location##NAME >= 0) {                                           \
               FUNCTION_ARB;                                                     \
            }

#define  SET_UNIFORM_FLOAT(PROGRAM, NAME, VALUE)   \
            CALL_ARB_FUNCTION(PROGRAM, NAME, glUniform1fARB(location##NAME, VALUE) );         


#define  SET_UNIFORM_VECTOR(PROGRAM, NAME, VALUE)  \
            CALL_ARB_FUNCTION(PROGRAM, NAME, glUniform4fvARB(location##NAME, 1, VALUE) );         


#define  SET_UNIFORM_TEXTURE(PROGRAM, NAME, VALUE, TYPE) \
            CALL_ARB_FUNCTION(PROGRAM, NAME, setUniformTexture(location##NAME, VALUE, TYPE, 0); );         


bool  checkGL();

bool  getLastGLError();

bool  loadShader(GLhandleARB shader, const char * fileName);

bool  createGLTexture   (const char* name, GLuint* handle);
bool  createGLTexture3D (const char* name, GLuint* handle);

bool  createGLCubeMap   (const std::string& name, GLuint* handle);

bool  createShaders     (const char* vshName, GLhandleARB* vertexShader, const char* fshName, GLhandleARB* fragmentShader, GLhandleARB* program);

// these are slow versions, as they always request location:
bool  setUniformVector  ( GLhandleARB program, const char * name, const float * value );
bool  setUniformFloat   ( GLhandleARB program, const char * name, float value );

bool  setUniformTexture (GLhandleARB location, GLuint textureObject, int type, int unitNumber = 0);

void  renderBuffers     (GLuint posVBO, GLuint normalVBO, int vertexNumber);

bool  createVBO         (GLuint* vbo, unsigned int size);
void  deleteVBO         (GLuint* vbo);


#endif //_GL_UTILS_H_