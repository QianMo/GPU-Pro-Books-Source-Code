#ifndef OGL_INDEX_BUFFER_H
#define OGL_INDEX_BUFFER_H

// OGL_IndexBuffer
//
// Manages an index buffer.  
class OGL_IndexBuffer
{
public:
  OGL_IndexBuffer():
    indexBufferName(0),
    indices(NULL),
    currentIndexCount(0),
    maxIndexCount(0),
    dynamic(false)
  {
  }

  ~OGL_IndexBuffer()
  {
    Release();
  }

  void Release();

  bool Create(unsigned int maxIndexCount, bool dynamic);

  void Clear()
  {
    currentIndexCount = 0;
  }

  unsigned int AddIndices(unsigned int numIndices, const unsigned int *newIndices);

  bool Update();

  void Bind() const;

  unsigned int GetIndexCount() const
  {
    return currentIndexCount;
  }

  bool IsDynamic() const
  {
    return dynamic;
  }

private:
  GLuint indexBufferName; 
  unsigned int *indices; 
  unsigned int currentIndexCount; // current count of indices
  unsigned int maxIndexCount; // max count of indices that index buffer can handle
  bool dynamic;

};

#endif