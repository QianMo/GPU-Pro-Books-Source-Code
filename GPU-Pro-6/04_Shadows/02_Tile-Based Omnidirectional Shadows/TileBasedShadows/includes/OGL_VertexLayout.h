#ifndef OGL_VERTEX_LAYOUT_H
#define OGL_VERTEX_LAYOUT_H

#include <vertex_types.h>

// OGL_VertexLayout
//
class OGL_VertexLayout
{
public:
  OGL_VertexLayout():
    vertexArrayName(0),
    vertexElementDescs(NULL),
    numVertexElementDescs(0)
  {
  }

  ~OGL_VertexLayout()
  {
    Release();
  }

  void Release();

  bool Create(const VertexElementDesc *vertexElementDescs, unsigned int numVertexElementDescs);

  void Bind() const;

  unsigned int CalcVertexSize() const;

  bool IsEqual(const VertexElementDesc *vertexElementDescs, unsigned int numVertexElementDescs) const;

private: 
  GLuint vertexArrayName; 
  VertexElementDesc *vertexElementDescs;
  unsigned int numVertexElementDescs;

};

#endif