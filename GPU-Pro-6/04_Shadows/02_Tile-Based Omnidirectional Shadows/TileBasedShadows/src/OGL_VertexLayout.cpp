#include <stdafx.h>
#include <Demo.h>
#include <OGL_VertexLayout.h>
 
static const GLint elementFormatCounts[] = { 1, 2, 3, 4 };
static const GLenum elementFormatTypes[] = { GL_FLOAT, GL_FLOAT, GL_FLOAT, GL_FLOAT };
static const unsigned int elementFormatSizes[] = { 4, 8, 12, 16 };

void OGL_VertexLayout::Release()
{
  if(vertexArrayName > 0)
    glDeleteVertexArrays(1, &vertexArrayName);
  SAFE_DELETE_ARRAY(vertexElementDescs);
}

bool OGL_VertexLayout::Create(const VertexElementDesc *vertexElementDescs, unsigned int numVertexElementDescs)
{	
  if((!vertexElementDescs) || (numVertexElementDescs < 1))
    return false;

  this->numVertexElementDescs = numVertexElementDescs;
  this->vertexElementDescs = new VertexElementDesc[numVertexElementDescs];
  if(!this->vertexElementDescs)
    return false;
  memcpy(this->vertexElementDescs, vertexElementDescs, sizeof(VertexElementDesc)*numVertexElementDescs);

  glGenVertexArrays(1, &vertexArrayName);
  glBindVertexArray(vertexArrayName);
  for(unsigned int i=0; i<numVertexElementDescs; i++)
  {
    glVertexAttribFormat(vertexElementDescs[i].location, elementFormatCounts[vertexElementDescs[i].format], 
                         elementFormatTypes[vertexElementDescs[i].format], GL_FALSE, vertexElementDescs[i].offset);
    glVertexAttribBinding(vertexElementDescs[i].location, 0);
    glEnableVertexAttribArray(vertexElementDescs[i].location);
  }
  glBindVertexArray(0);

  return true;
}

void OGL_VertexLayout::Bind() const
{
  glBindVertexArray(vertexArrayName);  
}

unsigned int OGL_VertexLayout::CalcVertexSize() const
{
  unsigned int vertexSize = 0;
  for(unsigned int i=0; i<numVertexElementDescs; i++)
    vertexSize += elementFormatSizes[vertexElementDescs[i].format];
  return vertexSize;
}

bool OGL_VertexLayout::IsEqual(const VertexElementDesc *vertexElementDescs, unsigned int numVertexElementDescs) const
{
  if((!vertexElementDescs) || (this->numVertexElementDescs != numVertexElementDescs))
    return false;
  for(unsigned int i=0; i<numVertexElementDescs; i++)
  {
    if(this->vertexElementDescs[i].location != vertexElementDescs[i].location)
      return false;
    if(this->vertexElementDescs[i].format != vertexElementDescs[i].format)
      return false;
    if(this->vertexElementDescs[i].offset != vertexElementDescs[i].offset)
      return false;
  }
  return true;
}