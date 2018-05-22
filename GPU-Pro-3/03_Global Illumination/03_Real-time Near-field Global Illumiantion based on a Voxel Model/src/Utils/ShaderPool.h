#ifndef SHADERPOOL_H
#define SHADERPOOL_H

#include "OpenGL.h"
#include "ShaderProgram.h"

class ShaderPool
{
public:
   static ShaderProgram* getQuad();      ///< outputs a 2D texture (all 4 channels)
   static ShaderProgram* getQuad2Tex();  ///< adds or multiplies two 2D textures
   static ShaderProgram* getQuadCombine();
   static ShaderProgram* getQuadGamma();

   static ShaderProgram* getWriteColorRGB(float r = 0, float g = 0, float b = 0);  ///< outputs a colored quad
   static ShaderProgram* getWriteContribTex();

   static ShaderProgram* getQuadClampToZero();

   static ShaderProgram* getCreateVoxelRaysShaderProgram();


};


#endif
