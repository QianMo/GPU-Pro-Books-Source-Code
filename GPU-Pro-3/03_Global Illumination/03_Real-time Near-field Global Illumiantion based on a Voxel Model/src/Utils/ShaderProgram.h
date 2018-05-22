#ifndef SHADERPROGRAM_H
#define SHADERPROGRAM_H

#include "OpenGL.h"

#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <map>

using namespace std;


class ShaderProgram
{

public:

   ShaderProgram(string vertexShaderFile, string fragmentShaderFile);
   ShaderProgram(string vertexShaderFile, string geometryShaderFile, 
          GLenum inputType, GLenum outputType, GLuint maxPointsToEmit,
          string fragmentShaderFile);

   void useProgram();

   GLuint getProgram() const { return mProgram; }

   GLint getUniformLocation(string uniformName);


private:
   ShaderProgram();

   string readFile(string filename);
   void printShaderInfoLog(GLuint shader);
   void printProgramInfoLog(GLuint program);

   void createVertexShader(string filename);
   void createGeometryShader(string filename);
   void createFragmentShader(string filename);

   /// Create program from vertex and fragment shader objects
   void createProgram();

   /// Create program from vertex, geometry and fragment shader objects
   /// with parameter settings for geometry shader
   void createProgram(GLenum inputType, GLenum outputType, GLuint maxPointsToEmit);

   GLuint mVertexShader;
   GLuint mGeometryShader;
   GLuint mFragmentShader;
   GLuint mProgram;

   // Settings for geometry shader
   GLenum mInputType;
   GLenum mOutputType;
   GLuint mMaxPointsToEmit;

   map<string,GLint> mUniformLocations;

};

#endif 
