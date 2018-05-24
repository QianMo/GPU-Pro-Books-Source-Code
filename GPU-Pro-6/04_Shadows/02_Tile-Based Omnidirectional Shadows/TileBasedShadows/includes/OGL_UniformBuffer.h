#ifndef OGL_UNIFORM_BUFFER_H 
#define OGL_UNIFORM_BUFFER_H

#include <render_states.h>

// OGL_UniformBuffer
//
class OGL_UniformBuffer
{
public:
  OGL_UniformBuffer(): 
    uniformBufferName(0),
    size(0)
  {
  }

  ~OGL_UniformBuffer()
  {
    Release();
  }

  void Release();

  bool Create(unsigned int bufferSize);

  // Please note: uniforms must be aligned according to the GLSL std140 rules, in order to be able
  // to upload data in 1 block.
  bool Update(const void *uniformBufferData);

  void Bind(uniformBufferBP bindingPoint) const;

private:
  GLuint uniformBufferName;
  unsigned int size; // size of uniform data

};

#endif