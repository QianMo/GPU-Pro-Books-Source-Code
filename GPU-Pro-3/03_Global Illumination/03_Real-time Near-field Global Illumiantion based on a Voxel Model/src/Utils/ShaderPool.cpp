#include "ShaderPool.h"

ShaderProgram* ShaderPool::getQuad()
{
   static ShaderProgram* pQuad = new ShaderProgram("src/shader/Quad.vert", "src/shader/Quad.frag");
   pQuad->useProgram();
   glUniform1i(pQuad->getUniformLocation("tex"), 0);
   return pQuad;
}
ShaderProgram* ShaderPool::getWriteContribTex()
{
   static ShaderProgram* pContr = new ShaderProgram("src/shader/Quad.vert", "src/shader/WriteContrib.frag");
   pContr->useProgram();
   glUniform1i(pContr->getUniformLocation("inputTex"), 0);
   return pContr;
}

ShaderProgram* ShaderPool::getQuadGamma()
{
   static ShaderProgram* pQuad = new ShaderProgram("src/shader/Quad.vert", "src/shader/QuadGamma.frag");
   pQuad->useProgram();
   glUniform1i(pQuad->getUniformLocation("tex"), 0);
   return pQuad;
}

ShaderProgram* ShaderPool::getQuad2Tex() // Multiply or add two textures
{
   static ShaderProgram* pQuad2 = new ShaderProgram("src/shader/Quad.vert", "src/shader/Quad2Tex.frag");
   pQuad2->useProgram();
   glUniform1i(pQuad2->getUniformLocation("tex0"), 0);
   glUniform1i(pQuad2->getUniformLocation("tex1"), 1);
   glUniform1f(pQuad2->getUniformLocation("factor"), 1.0f);
   return pQuad2;
}


ShaderProgram* ShaderPool::getQuadCombine()
{
   static ShaderProgram* p = new ShaderProgram("src/shader/Quad.vert", "src/shader/QuadCombine.frag");
   p->useProgram();
   glUniform1i(p->getUniformLocation("directLightBuffer"), 0);
   glUniform1i(p->getUniformLocation("indirectLightBuffer"), 1);
   glUniform1i(p->getUniformLocation("materialBuffer"), 2);
   glUniform1i(p->getUniformLocation("positionBuffer"), 3); 
   return p;
}

ShaderProgram* ShaderPool::getQuadClampToZero()
{
   static ShaderProgram* p = new ShaderProgram("src/shader/Quad.vert", "src/shader/ClampToZero.frag");
   p->useProgram();
   glUniform1i(p->getUniformLocation("input"), 0);
   return p;

}

ShaderProgram* ShaderPool::getWriteColorRGB(float r, float g, float b)
{
   static ShaderProgram* p = new ShaderProgram("src/shader/Quad.vert", "src/shader/WriteColorRGB.frag");
   p->useProgram();
   glUniform3f(p->getUniformLocation("color"), r, g, b);
   return p;
}

ShaderProgram* ShaderPool::getCreateVoxelRaysShaderProgram()
{
   static ShaderProgram* rS = new ShaderProgram("src/shader/Quad.vert", "src/shader/CreateVoxelRays.frag");
   return rS;
}