#include <stdafx.h>
#include <Demo.h>
#include <DX12_Shader.h>

static const char* shaderModels[NUM_SHADER_TYPES] =
  { "vs_5_1", "hs_5_1", "ds_5_1", "gs_5_1", "ps_5_1", "cs_5_1" };

static bool ReadShaderFile(const char *fileName, unsigned char **data, UINT *dataSize)
{
	if((!fileName) || (!data) || (!dataSize))
		return false;
	char filePath[DEMO_MAX_FILEPATH];
	if(!Demo::fileManager->GetFilePath(fileName, filePath))
		return false;
	FILE *file;
	fopen_s(&file, filePath, "rt");
	if(!file)
		return false;
	struct _stat fileStats;
	if(_stat(filePath, &fileStats) != 0)
	{
		fclose(file);
		return false;
	}
	 *data = new unsigned char[fileStats.st_size + 1];
	if(!(*data))
	{
		fclose(file);
		return false;
	}
	*dataSize = static_cast<UINT>(fread(*data, 1, fileStats.st_size, file));
	(*data)[*dataSize] = 0;
	fclose(file);
	return true;
}

class ShaderInclude: public ID3DInclude
{
public:
	STDMETHOD(Open)(D3D_INCLUDE_TYPE IncludeType, LPCSTR pFileName, LPCVOID pParentData, LPCVOID *ppData, UINT *pBytes) 
	{
		char fileName[DEMO_MAX_FILENAME];
		strcpy(fileName, "shaders/HLSL/");
		strcat(fileName, pFileName);
		if(!ReadShaderFile(fileName, (unsigned char**)ppData, pBytes))
			return S_FALSE;    
		return S_OK;
	}

	STDMETHOD(Close)(LPCVOID pData) 
	{
		SAFE_DELETE_ARRAY(pData);
		return S_OK;
	}
} includeHandler;


void DX12_Shader::LoadDefines(std::ifstream &file)
{
	std::string str;
	file >> str;        
	while(true)
	{
		file >> str;
		if((str == "}") || (file.eof()))
			break;
		assert(str.length() <= DEMO_MAX_STRING);
		char elementString[DEMO_MAX_STRING];
		strcpy(elementString, str.c_str());
		defineStrings.AddElement(&elementString);
	}
}

bool DX12_Shader::CreateShaderMacros()
{
	for(UINT i=0; i<defineStrings.GetSize(); i++)
	{
    if(permutationMask & (1<<i))
		{
			D3D_SHADER_MACRO shaderMacro;
			shaderMacro.Name = defineStrings[i];
			if(!shaderMacro.Name)
				return false;
			shaderMacro.Definition = nullptr;
			shaderMacros.AddElement(&shaderMacro);
		}
	}
	D3D_SHADER_MACRO shaderMacro;
	shaderMacro.Name = nullptr;
	shaderMacro.Definition = nullptr;
	shaderMacros.AddElement(&shaderMacro);
	return true;
}

bool DX12_Shader::LoadShaderUnit(shaderTypes shaderType, std::ifstream &file)
{
  DWORD dwShaderFlags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifdef _DEBUG
	dwShaderFlags |= D3DCOMPILE_DEBUG;
#endif

	std::string str, token;
	file >> token >> str >> token;
	if(str == "nullptr")
    return false;
	
	std::string filename = "shaders/HLSL/";
	filename.append(str);

  // read shader file
  unsigned char *shaderSource = nullptr;
  UINT shaderSize = 0;
  if(!ReadShaderFile(filename.c_str(), &shaderSource, &shaderSize))
		return false; 

  // compile shader
  ComPtr<ID3DBlob> errorMsg;
  if(FAILED(D3DCompile(shaderSource, shaderSize, nullptr, &shaderMacros[0], &includeHandler, "main",
	    shaderModels[shaderType], dwShaderFlags, 0, &byteCodes[shaderType], &errorMsg)))
  {
	  if(errorMsg)
	  {
		  char errorTitle[512];
		  wsprintf(errorTitle, "Shader Compile Error in %s", filename.c_str());
		  MessageBox(nullptr, reinterpret_cast<char*>(errorMsg->GetBufferPointer()), errorTitle, MB_OK | MB_ICONEXCLAMATION);
	  }
    SAFE_DELETE_ARRAY(shaderSource);
	  return false;
  } 
  SAFE_DELETE_ARRAY(shaderSource);
	
  shaderUnitMask |= (1 << shaderType);

	return true;
}

bool DX12_Shader::Load(const char *fileName, UINT permutationMask)
{
	strcpy(name, fileName);
	this->permutationMask = permutationMask;

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
			if(!CreateShaderMacros())
			{
				file.close();
				return false;
			}
		}
		else if(str == "VertexShader")
		{
			if(!LoadShaderUnit(VERTEX_SHADER, file))
			{
				file.close();
				return false;
			}
		}
    else if(str == "HullShader")
    {
      if (!LoadShaderUnit(HULL_SHADER, file))
      {
        file.close();
        return false;
      }
    }
    else if (str == "DomainShader")
    {
      if (!LoadShaderUnit(DOMAIN_SHADER, file))
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
		else if(str == "PixelShader")
		{
			if(!LoadShaderUnit(PIXEL_SHADER, file))
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
	
	shaderMacros.Erase();
  defineStrings.Erase();

	return true;
}

D3D12_SHADER_BYTECODE DX12_Shader::GetByteCode(shaderTypes shaderType)
{
  D3D12_SHADER_BYTECODE shaderByteCode;
  if (shaderUnitMask & (1 << shaderType))
  {
    shaderByteCode.pShaderBytecode = byteCodes[shaderType]->GetBufferPointer();
    shaderByteCode.BytecodeLength = byteCodes[shaderType]->GetBufferSize();
  }
  else
  {
    shaderByteCode.pShaderBytecode = nullptr;
    shaderByteCode.BytecodeLength = 0;
  }
  return shaderByteCode;
}
