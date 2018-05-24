#include <stdafx.h>
#include <Demo.h>
#include <OGL_StructuredBuffer.h>

void OGL_StructuredBuffer::Release()
{
  if(structuredBufferName > 0)
    glDeleteBuffers(1, &structuredBufferName);
}

bool OGL_StructuredBuffer::Create(unsigned int elementCount, unsigned int elementSize, unsigned int flags)
{
  this->elementCount = elementCount;
  this->elementSize = elementSize;
  this->flags = flags;
  const bool append = flags & APPEND_SBF;
  const bool indirectDraw = flags & INDIRECT_DRAW_SBF;
  if((!append) && (indirectDraw))
    return false;

  unsigned int bufferSize = elementCount*elementSize;

  // add space for atomic counter
  if(append)
    bufferSize += 4;

  glGenBuffers(1, &structuredBufferName);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, structuredBufferName);
  GLenum bufferType = (flags & DYNAMIC_SBF) ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW;  
  glBufferData(GL_SHADER_STORAGE_BUFFER, bufferSize, NULL, bufferType);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

  return true;
}

void OGL_StructuredBuffer::BeginUpdate()
{
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, structuredBufferName);
  mappedStructuredBuffer = (unsigned char*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
}

bool OGL_StructuredBuffer::Update(unsigned int elementIndex, unsigned int offset, unsigned int elementDataSize, const void *elementData)
{
  if((elementIndex >= elementCount) || (elementDataSize < 1) || (!elementData) || (!mappedStructuredBuffer))
    return false;
  memcpy(&mappedStructuredBuffer[elementIndex*elementSize+offset], elementData, elementDataSize);
  return true;
}

void OGL_StructuredBuffer::EndUpdate()
{
  glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
  mappedStructuredBuffer = NULL; 
}

void OGL_StructuredBuffer::BindToRenderTarget(unsigned int currentBindingPoint)
{
  // clear atomic counter
  if(flags & APPEND_SBF)
  {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, structuredBufferName);
    GLuint clearData = 0;
    glClearBufferSubData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, 0, sizeof(GLuint), GL_LUMINANCE, GL_UNSIGNED_INT, &clearData);
  }

  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, currentBindingPoint, structuredBufferName);
}

void OGL_StructuredBuffer::Bind(structuredBufferBP bindingPoint) const
{
  if(flags & INDIRECT_DRAW_SBF)
  {
    glBindBuffer(GL_PARAMETER_BUFFER_ARB, structuredBufferName);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, structuredBufferName);  
  }
  else
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bindingPoint, structuredBufferName);
}

