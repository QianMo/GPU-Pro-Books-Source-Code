#include <stdafx.h>
#include <Demo.h>
#include <OGL_VertexBuffer.h>

void OGL_VertexBuffer::Release()
{
  if(vertexBufferName > 0)
    glDeleteBuffers(1, &vertexBufferName);
  SAFE_DELETE_ARRAY(vertices);
}

bool OGL_VertexBuffer::Create(unsigned int vertexSize, unsigned int maxVertexCount, bool dynamic)
{
  if((vertexSize < 1) || (maxVertexCount < 1))
    return false;

  this->vertexSize = vertexSize;
  this->maxVertexCount = maxVertexCount;
  this->dynamic = dynamic;
  vertices = new char[vertexSize*maxVertexCount];
  if(!vertices)
    return false;

  glGenBuffers(1, &vertexBufferName);
  glBindBuffer(GL_ARRAY_BUFFER, vertexBufferName);
  unsigned int arrayObjectSize = vertexSize*maxVertexCount; 
  GLenum VBOType = dynamic ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW;
  glBufferData(GL_ARRAY_BUFFER, arrayObjectSize, NULL, VBOType); 
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  return true;
}

unsigned int OGL_VertexBuffer::AddVertices(unsigned int numVertices, const void *newVertices)
{
  int firstIndex = currentVertexCount;
  currentVertexCount += numVertices;
  assert(currentVertexCount <= maxVertexCount);
  memcpy(&vertices[vertexSize*firstIndex], newVertices, vertexSize*numVertices);
  return firstIndex;
}

bool OGL_VertexBuffer::Update()
{
  if(currentVertexCount > 0)
  {
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferName);
    GLvoid* buffer = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    memcpy(buffer, vertices, vertexSize*currentVertexCount);
    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    if(dynamic)
      Clear();
  }
  return true;
}

void OGL_VertexBuffer::Bind() const
{
  glBindVertexBuffer(0, vertexBufferName, 0, vertexSize);
}
