#ifndef OGL_SHADER_H
#define OGL_SHADER_H

#include <List.h>
#include <render_states.h>

class OGL_UniformBuffer; 
class OGL_StructuredBuffer;
class OGL_Texture;
class OGL_Sampler;

// OGL_Shader
//
// Loaded from a simple text-file (".sdr"), that references the actual shader source files.
class OGL_Shader
{
public:	
  OGL_Shader():
    permutationMask(0), 
    incSource(NULL),
    shaderProgramName(0),
    vertexShaderName(0),
    geometryShaderName(0),
    fragmentShaderName(0),
    computeShaderName(0)
  {  
    name[0] = 0;
  }

  ~OGL_Shader()
  {
    Release();
  }

  void Release();

  bool Load(const char *fileName, unsigned int permutationMask=0);

  void Bind() const;	
  
  void SetTexture(textureBP bindingPoint, const OGL_Texture *texture, const OGL_Sampler *sampler=NULL) const;

  void SetUniformBuffer(uniformBufferBP bindingPoint, const OGL_UniformBuffer *uniformBuffer) const;

  void SetStructuredBuffer(structuredBufferBP bindingPoint, const OGL_StructuredBuffer *structuredBuffer) const;

  unsigned int GetPermutationMask() const
  {
    return permutationMask;
  }

  const char* GetName() const
  {
    return name;
  }

private:
  void LoadDefines(std::ifstream &file);

  const char* ReadShaderFile(const char *fileName);

  bool InitShaderUnit(shaderTypes shaderType, const char *fileName);

  bool LoadShaderUnit(shaderTypes shaderType, std::ifstream &file);

  bool InitProgram();

  char name[DEMO_MAX_FILENAME];
  unsigned int permutationMask;
  const char *incSource;
  List<char[DEMO_MAX_STRING]> defineStrings;
  
  GLuint shaderProgramName;
  GLuint vertexShaderName;
  GLuint geometryShaderName; 
  GLuint fragmentShaderName; 
  GLuint computeShaderName;
  
};

#endif
