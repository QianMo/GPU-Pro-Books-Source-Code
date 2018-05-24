#include <stdafx.h>
#include <Demo.h>
#include <OGL_Shader.h>

void OGL_Shader::Release()
{
  if(shaderProgramName > 0)
    glDeleteProgram(shaderProgramName);
  SAFE_DELETE_ARRAY(incSource);
}

void OGL_Shader::LoadDefines(std::ifstream &file)
{
  std::string str, defineString;
  file >> str;        
  while(true)
  {
    file >> str;
    if((str == "}") || (file.eof()))
      break;
    defineString.clear();
    defineString += "#define ";
    defineString += str;
    defineString += "\n";	
    assert(str.length() <= DEMO_MAX_STRING);
    char elementString[DEMO_MAX_STRING];
    strcpy(elementString, defineString.c_str());
    defineStrings.AddElement(&elementString);
  }
}

const char* OGL_Shader::ReadShaderFile(const char *fileName)
{
  if(!fileName)
    return NULL;
  char filePath[DEMO_MAX_FILEPATH];
  if(!Demo::fileManager->GetFilePath(fileName, filePath))
    return NULL;
  FILE *file;
  fopen_s(&file, filePath, "rt");
  if(!file)
    return NULL;
  struct _stat fileStats;
  if(_stat(filePath, &fileStats) != 0)
  {
    fclose(file);
    return NULL;
  }
  unsigned char *buffer = new unsigned char[fileStats.st_size + 1];
  if(!buffer)
  {
    fclose(file);
    return NULL;
  }
  unsigned int bytes = fread(buffer, 1, fileStats.st_size, file);
  buffer[bytes] = 0;
  fclose(file);
  return (const char*)buffer;
}

bool OGL_Shader::InitShaderUnit(shaderTypes shaderType, const char *fileName)
{
  char profileString[256];
  sprintf(profileString, "#version 440 core\n#extension GL_ARB_shader_draw_parameters: enable\nlayout(std140, column_major) uniform;\nlayout(std430, column_major) buffer;\n");
  if(Demo::renderer->IsGpuAMD())
    strcat(profileString, "#define AMD_GPU\n");
  GLuint shaderName;
  const char* shaderStrings[4]; 
  shaderStrings[0] = profileString;
  shaderStrings[1] = incSource;
  std::string permutationString;
  for(unsigned int i=0; i<defineStrings.GetSize(); i++)
  {
    if(permutationMask & (1<<i))
      permutationString += defineStrings[i];
  }
  shaderStrings[2] = permutationString.c_str();
  switch(shaderType)
  {
  case VERTEX_SHADER:
    shaderName = glCreateShader(GL_VERTEX_SHADER);
    vertexShaderName = shaderName;
    shaderStrings[3] = ReadShaderFile(fileName);
    break;

  case GEOMETRY_SHADER:
    shaderName = glCreateShader(GL_GEOMETRY_SHADER);
    geometryShaderName = shaderName;
    shaderStrings[3] = ReadShaderFile(fileName);
    break;

  case FRAGMENT_SHADER:
    shaderName = glCreateShader(GL_FRAGMENT_SHADER);
    fragmentShaderName = shaderName;
    shaderStrings[3] = ReadShaderFile(fileName);
    break;

  case COMPUTE_SHADER:
    shaderName = glCreateShader(GL_COMPUTE_SHADER);
    computeShaderName = shaderName;
    shaderStrings[3] = ReadShaderFile(fileName);
    break;
  }		
  glShaderSource(shaderName, 4, shaderStrings, NULL);
  glCompileShader(shaderName);
  SAFE_DELETE_ARRAY(shaderStrings[3]);
  GLint shaderCompiled;
  glGetShaderiv(shaderName, GL_COMPILE_STATUS, &shaderCompiled);
  if(!shaderCompiled)
  {
    char errorTitle[512];
    _snprintf(errorTitle, sizeof(errorTitle), "Shader Compile Error in %s", fileName);
    char errorMsg[4096];
    glGetShaderInfoLog(shaderName, sizeof(errorMsg), NULL, errorMsg);
    MessageBox(NULL, errorMsg, errorTitle, MB_OK | MB_ICONEXCLAMATION);
    return false;
  }
  return true;
}

bool OGL_Shader::LoadShaderUnit(shaderTypes shaderType, std::ifstream &file)
{
  std::string str, token; 
  file >> token >> str >> token;
  if(str != "NULL")
  {
    std::string filename = "shaders/GLSL/";
    filename.append(str);
    if(!InitShaderUnit(shaderType, filename.c_str()))
    {
      file.close();
      Release(); 
      return false; 
    }
  }
  return true;
}

bool OGL_Shader::InitProgram()
{
  shaderProgramName = glCreateProgram();
  if(vertexShaderName > 0)
  {
    glAttachShader(shaderProgramName, vertexShaderName); 
    glDeleteShader(vertexShaderName);
  }
  if(geometryShaderName > 0)
  {
    glAttachShader(shaderProgramName, geometryShaderName);
    glDeleteShader(geometryShaderName);
  }
  if(fragmentShaderName > 0)
  {
    glAttachShader(shaderProgramName, fragmentShaderName);
    glDeleteShader(fragmentShaderName);
  }
  if(computeShaderName > 0)
  {
    glAttachShader(shaderProgramName, computeShaderName);
    glDeleteShader(computeShaderName);
  } 
  glLinkProgram(shaderProgramName);
  GLint linked;
  glGetProgramiv(shaderProgramName, GL_LINK_STATUS, &linked);
  if(!linked)
  {
    char errorTitle[512];
    _snprintf(errorTitle, sizeof(errorTitle), "Linking Error in %s", name);
    char errorMsg[4096];	
    glGetProgramInfoLog(shaderProgramName, sizeof(errorMsg), NULL, errorMsg);
    MessageBox(NULL, errorMsg, errorTitle, MB_OK | MB_ICONEXCLAMATION);
    return false;
  }
  return true;
}

bool OGL_Shader::Load(const char *fileName, unsigned int permutationMask)
{
  strcpy(name, fileName);
  this->permutationMask = permutationMask;

  incSource = ReadShaderFile("shaders/GLSL/globals.shi");

  char filePath[DEMO_MAX_FILEPATH];
  if(!Demo::fileManager->GetFilePath(fileName, filePath))
    return false;
  std::ifstream file(filePath, std::ios::in);
  if(!file.is_open())
    return false;

  std::string str, token;
  file >> str; 
  while(!file.eof())
  {
    if(str == "Defines")
    {
      LoadDefines(file);
    }
    else if(str == "VertexShader")
    {
      if(!LoadShaderUnit(VERTEX_SHADER, file))
      {
        file.close();
        return false;
      }
    }
    else if(str == "GeometryShader")
    {
      if(!LoadShaderUnit(GEOMETRY_SHADER, file))
      {
        file.close();
        return false;
      }
    }
    else if(str == "FragmentShader")
    {
      if(!LoadShaderUnit(FRAGMENT_SHADER, file))
      {
        file.close();
        return false;
      }
    }
    else if(str == "ComputeShader")
    {
      if(!LoadShaderUnit(COMPUTE_SHADER, file))
      {
        file.close();
        return false;
      }
    }
    file >> str;
  } 
  file.close();

  if(!InitProgram())
  {
    Release(); 
    return false;
  }

  SAFE_DELETE_ARRAY(incSource);

  return true;
}

void OGL_Shader::Bind() const
{
  glUseProgram(shaderProgramName);
}

void OGL_Shader::SetTexture(textureBP bindingPoint, const OGL_Texture *texture, const OGL_Sampler *sampler) const
{
  if(texture)
  {
    texture->Bind(bindingPoint);
    if(sampler)
      sampler->Bind(bindingPoint);
    else
      Demo::renderer->GetSampler(POINT_SAMPLER_ID)->Bind(bindingPoint);
  }
}

void OGL_Shader::SetUniformBuffer(uniformBufferBP bindingPoint, const OGL_UniformBuffer *uniformBuffer) const
{
  if(uniformBuffer) 
    uniformBuffer->Bind(bindingPoint); 
}

void OGL_Shader::SetStructuredBuffer(structuredBufferBP bindingPoint, const OGL_StructuredBuffer *structuredBuffer) const
{
  if(structuredBuffer)
    structuredBuffer->Bind(bindingPoint);
}







