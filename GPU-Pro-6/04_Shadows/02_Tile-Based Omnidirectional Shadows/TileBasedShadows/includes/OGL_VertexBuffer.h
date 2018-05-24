#ifndef OGL_VERTEX_BUFFER_H
#define OGL_VERTEX_BUFFER_H

#include <vertex_types.h>

// OGL_VertexBuffer
//
class OGL_VertexBuffer
{
public:
  OGL_VertexBuffer():
    vertexBufferName(0),
    vertices(NULL),
    vertexSize(0),
    currentVertexCount(0),
    maxVertexCount(0),
    dynamic(false)
  {
  }

  ~OGL_VertexBuffer()
  {
    Release();
  }

  void Release();

  bool Create(unsigned int vertexSize, unsigned int maxVertexCount, bool dynamic);

  void Clear()
  {
    currentVertexCount = 0;
  }

  unsigned int AddVertices(unsigned int numVertices, const void *newVertices);

  bool Update();

  void Bind() const;

  unsigned int GetVertexSize() const
  {
    return vertexSize;
  }

  unsigned int GetVertexCount() const
  {
    return currentVertexCount;
  }

  bool IsDynamic() const
  {
    return dynamic;
  }

private: 
  GLuint vertexBufferName; 
  char *vertices;
  unsigned int vertexSize; // size of 1 vertex
  unsigned int currentVertexCount; // current count of vertices 
  unsigned int maxVertexCount; // max count of vertices that vertex buffer can handle
  bool dynamic;
 
};

#endif