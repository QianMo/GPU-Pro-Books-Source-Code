#include <stdafx.h>
#include <Demo.h>
#include <OGL_UniformBuffer.h>

void OGL_UniformBuffer::Release()
{
  if(uniformBufferName > 0)
    glDeleteBuffers(1, &uniformBufferName);
}

bool OGL_UniformBuffer::Create(unsigned int bufferSize)
{
  if(bufferSize < 4)
    return false;

  size = bufferSize;
  unsigned int align = bufferSize % 16;
  if(align > 0)
    bufferSize += 16-align;

  glGenBuffers(1, &uniformBufferName);	
  glBindBuffer(GL_UNIFORM_BUFFER, uniformBufferName);
  glBufferData(GL_UNIFORM_BUFFER, bufferSize, NULL, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_UNIFORM_BUFFER, 0); 

  return true;
}

bool OGL_UniformBuffer::Update(const void *uniformBufferData) 
{
  if(!uniformBufferData)
    return false;

  glBindBuffer(GL_UNIFORM_BUFFER, uniformBufferName);
  void *mappedUniformBuffer = glMapBuffer(GL_UNIFORM_BUFFER, GL_WRITE_ONLY);
  memcpy(mappedUniformBuffer, uniformBufferData, size);
  glUnmapBuffer(GL_UNIFORM_BUFFER);
  glBindBuffer(GL_UNIFORM_BUFFER, 0);

  return true;
}

void OGL_UniformBuffer::Bind(uniformBufferBP bindingPoint) const
{
  glBindBufferBase(GL_UNIFORM_BUFFER, bindingPoint, uniformBufferName);
}

