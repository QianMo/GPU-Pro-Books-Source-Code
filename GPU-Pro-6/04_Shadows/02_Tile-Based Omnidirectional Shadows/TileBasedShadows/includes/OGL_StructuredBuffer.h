#ifndef OGL_STRUCTURED_BUFFER_H
#define OGL_STRUCTURED_BUFFER_H

#include <render_states.h>

enum structuredBufferFlags
{
  DYNAMIC_SBF=1, // dynamic updates
  APPEND_SBF=2, // append buffer
  INDIRECT_DRAW_SBF=4 // indirect draw buffer
};

// OGL_StructuredBuffer
//
class OGL_StructuredBuffer
{
public:
  OGL_StructuredBuffer():   
    structuredBufferName(0),
    elementCount(0),
    elementSize(0),
    flags(0),
    mappedStructuredBuffer(NULL)
  {
  }

  ~OGL_StructuredBuffer()
  {
    Release();
  }

  void Release();

  bool Create(unsigned int elementCount, unsigned int elementSize, unsigned int flags=0);
   
  void BeginUpdate();

  bool Update(unsigned int elementIndex, unsigned int offset, unsigned int elementDataSize, const void *elementData);

  void EndUpdate();

  void BindToRenderTarget(unsigned int currentBindingPoint);

  void Bind(structuredBufferBP bindingPoint) const;

  unsigned int GetElementCount() const
  {
    return elementCount;
  }

  unsigned int GetElementSize() const
  {
    return elementCount;
  }

private:
  GLuint structuredBufferName;
  unsigned int elementCount; // number of structured elements in buffer
  unsigned int elementSize; // size of 1 structured element in bytes
  unsigned int flags;
  unsigned char* mappedStructuredBuffer;

};

#endif