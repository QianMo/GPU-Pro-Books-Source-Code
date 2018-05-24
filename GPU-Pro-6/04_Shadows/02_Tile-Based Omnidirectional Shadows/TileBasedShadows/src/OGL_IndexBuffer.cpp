#include <stdafx.h>
#include <Demo.h>
#include <OGL_IndexBuffer.h>

void OGL_IndexBuffer::Release()
{
  if(indexBufferName > 0)
    glDeleteBuffers(1, &indexBufferName);
  SAFE_DELETE_ARRAY(indices);
}

bool OGL_IndexBuffer::Create(unsigned int maxIndexCount, bool dynamic)
{
  this->maxIndexCount = maxIndexCount;
  this->dynamic = dynamic;
  indices = new unsigned int[maxIndexCount];
  if(!indices)
    return false;

  glGenBuffers(1, &indexBufferName);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBufferName);
  unsigned int arrayObjectSize = sizeof(unsigned int)*maxIndexCount;
  GLenum IBOType = dynamic ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW;
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, arrayObjectSize, NULL, IBOType); 
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  return true;
}

unsigned int OGL_IndexBuffer::AddIndices(unsigned int numIndices, const unsigned int *newIndices)
{
  int firstIndex = currentIndexCount;
  currentIndexCount += numIndices;
  assert(currentIndexCount <= maxIndexCount);
  memcpy(&indices[firstIndex], newIndices, sizeof(unsigned int)*numIndices);
  return firstIndex;
}

bool OGL_IndexBuffer::Update()
{
  if(currentIndexCount > 0)
  {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBufferName);
    GLvoid* buffer = glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY);
    memcpy(buffer, indices, sizeof(unsigned int)*currentIndexCount);
    glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    if(dynamic)
      Clear();
  }
  return true;
}

void OGL_IndexBuffer::Bind() const      
{
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBufferName);
}
