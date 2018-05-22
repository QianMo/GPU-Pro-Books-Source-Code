#include <stdafx.h>
#include <DEMO.h>
#include <DX11_SHADER.h>

const char* DX11_SHADER::shaderModels[NUM_SHADER_TYPES]=
  {"vs_5_0", "gs_5_0", "ps_5_0", "cs_5_0"};
const char* DX11_SHADER::uniformBufferRegisterNames[NUM_UNIFORM_BUFFER_BP] =
  {"b0", "b1", "b2"};
const char* DX11_SHADER::textureRegisterNames[NUM_TEXTURE_BP+NUM_STRUCTURED_BUFFER_BP] =
  {"t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10"};

class SHADER_INCLUDE: public ID3D10Include
{
public:
	STDMETHOD(Open)(D3D_INCLUDE_TYPE IncludeType, LPCSTR pFileName, LPCVOID pParentData, LPCVOID *ppData, UINT *pBytes) 
	{
		char fileName[DEMO_MAX_FILENAME];
		strcpy(fileName,"shaders/HLSL/src/");
		strcat(fileName,pFileName);
		if(!DX11_SHADER::ReadShaderFile(fileName,(unsigned char**)ppData,pBytes))
			return S_FALSE;    
		return S_OK;
	}

	STDMETHOD(Close)(LPCVOID pData) 
	{
		SAFE_DELETE_ARRAY(pData);
		return S_OK;
	}
} includeHandler;


void DX11_SHADER::Release()
{
	defineStrings.Erase();
	SAFE_RELEASE(vertexShader);
  SAFE_RELEASE(geometryShader);
	SAFE_RELEASE(fragmentShader);
	SAFE_RELEASE(computeShader);
	shaderMacros.Erase();
	for(int i=0;i<NUM_SHADER_TYPES;i++)
		SAFE_RELEASE(byteCodes[i]);
}

void DX11_SHADER::LoadDefines(std::ifstream &file)
{
	std::string str;
	file>>str;        
	while(true)
	{
		file>>str;
		if((str=="}")||(file.eof()))
			break;
		assert(str.length()<=DEMO_MAX_STRING);
		char elementString[DEMO_MAX_STRING];
		strcpy(elementString,str.c_str());
		defineStrings.AddElement(&elementString);
	}
}

bool DX11_SHADER::ReadShaderFile(const char *fileName,unsigned char **data,unsigned int *dataSize)
{
	if((!fileName)||(!data)||(!dataSize))
		return false;
	char filePath[DEMO_MAX_FILEPATH];
	if(!DEMO::fileManager->GetFilePath(fileName,filePath))
		return false;
	FILE *file;
	fopen_s(&file,filePath,"r");
	if(!file)
		return false;
	struct _stat fileStats;
	if(_stat(filePath,&fileStats)!=0)
	{
		fclose(file);
		return false;
	}
	*data = new unsigned char[fileStats.st_size];
	if(!(*data))
	{
		fclose(file);
		return false;
	}
	*dataSize = fread(*data,1,fileStats.st_size,file);
	(*data)[*dataSize] = 0;
	fclose(file);
	return true;
}

bool DX11_SHADER::CreateShaderUnit(shaderTypes shaderType,void *byteCode,int shaderSize)
{
	switch(shaderType)
	{
	case VERTEX_SHADER:
		if(!DEMO::renderer->GetDevice()->CreateVertexShader(byteCode,shaderSize,NULL,&vertexShader)==S_OK)
			return false;
		break;

	case GEOMETRY_SHADER:
		if(!DEMO::renderer->GetDevice()->CreateGeometryShader(byteCode,shaderSize,NULL,&geometryShader)==S_OK)
			return false;
		break;

	case FRAGMENT_SHADER:
		if(!DEMO::renderer->GetDevice()->CreatePixelShader(byteCode,shaderSize,NULL,&fragmentShader)==S_OK)
			return false;
		break;

	case COMPUTE_SHADER:
		if(!DEMO::renderer->GetDevice()->CreateComputeShader(byteCode,shaderSize,NULL,&computeShader)==S_OK)
			return false;
		break;
	}

	return true;
}

bool DX11_SHADER::InitShaderUnit(shaderTypes shaderType,const char *fileName)
{
	DWORD dwShaderFlags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifdef _DEBUG
	dwShaderFlags |= D3DCOMPILE_DEBUG;
#endif

	unsigned char *shaderSource = NULL;
	unsigned int shaderSize = 0;
	ID3D10Blob *errorMsgs = NULL;
	ID3D10Blob *shaderText = NULL;
	
	// read shader file
	if(!ReadShaderFile(fileName,&shaderSource,&shaderSize))
		return false;

	// preprocess shader
	if(D3DPreprocess(shaderSource,shaderSize,NULL,shaderMacros.entries,&includeHandler,&shaderText,&errorMsgs)!=S_OK)
	{
		if(errorMsgs)
		{
			char errorTitle[512];
			wsprintf(errorTitle,"Shader Preprocess Error in %s",fileName);
			MessageBox(NULL,(char*)errorMsgs->GetBufferPointer(),errorTitle,MB_OK|MB_ICONEXCLAMATION);
			SAFE_RELEASE(errorMsgs);
		}
		SAFE_DELETE_ARRAY(shaderSource);
		return false;
	}
	SAFE_RELEASE(errorMsgs);
	SAFE_DELETE_ARRAY(shaderSource);

	// parse shader
	ParseShaderString(shaderType,(const char*)shaderText->GetBufferPointer());

	// compile shader
	if(D3DCompile(shaderText->GetBufferPointer(),shaderText->GetBufferSize(),NULL,NULL,NULL,"main",
	   shaderModels[shaderType],dwShaderFlags,0,&byteCodes[shaderType],&errorMsgs)!=S_OK)
	{
		if(errorMsgs)
		{
			char errorTitle[512];
			wsprintf(errorTitle,"Shader Compile Error in %s",fileName);
			MessageBox(NULL,(char*)errorMsgs->GetBufferPointer(),errorTitle,MB_OK|MB_ICONEXCLAMATION);
			SAFE_RELEASE(errorMsgs);
		}
		SAFE_RELEASE(shaderText);
		return false;
	} 

	SAFE_RELEASE(shaderText);
	SAFE_RELEASE(errorMsgs);

	if(!CreateShaderUnit(shaderType,byteCodes[shaderType]->GetBufferPointer(),byteCodes[shaderType]->GetBufferSize()))
	{
		SAFE_RELEASE(byteCodes[shaderType]);
		return false;
	}
	
	return true;
}

bool DX11_SHADER::LoadShaderUnit(shaderTypes shaderType,std::ifstream &file)
{
	std::string str,token;
	file>>token>>str>>token;
	if(str!="NULL")
	{
		std::string filename = "shaders/HLSL/src/";
		filename.append(str);
		if(!InitShaderUnit(shaderType,filename.c_str()))
		{
			file.close();
			Release(); 
			return false; 
		}
	}
	return true;
}

void DX11_SHADER::ParseShaderString(shaderTypes shaderType,const char *shaderString)
{
	if(!shaderString)
		return;
	std::istringstream is(shaderString);
	std::string str,token,param;
	do
	{
		is>>str;
		if((str=="Texture1D")||(str=="Texture2D")||(str=="Texture2DMS")||
			 (str=="Texture3D")||(str=="TextureCube")||(str=="Texture2DArray")||
			 (str=="StructuredBuffer"))
		{
			is>>token;
			if(token=="<")
				is>>token>>token>>token;
      is>>token>>token>>token>>param;
			std::getline(is,token);
			for(int i=0;i<(NUM_TEXTURE_BP+NUM_STRUCTURED_BUFFER_BP);i++)
			{
				if(param==textureRegisterNames[i])
					textureMasks[i] |= 1<<shaderType;
			}			
		}
		else if(str=="cbuffer")
		{
			is>>token>>token>>token>>token>>param;
			std::getline(is,token);
			for(int i=0;i<NUM_UNIFORM_BUFFER_BP;i++)
			{
				if(param==uniformBufferRegisterNames[i])
					uniformBufferMasks[i] |= 1<<shaderType;
			}		
		}
		else if((str=="VS_OUTPUT")||(str=="void")||(str=="FS_OUTPUT"))
		{
			is>>str;
			std::getline(is,token);
		}
	}
	while((str!="main")||(is.eof()));
}

bool DX11_SHADER::CreateShaderMacros()
{
	for(int i=0;i<defineStrings.GetSize();i++)
	{
    if(permutationMask & (1<<i))
		{
			D3D10_SHADER_MACRO shaderMacro;
			shaderMacro.Name = defineStrings[i];
			if(!shaderMacro.Name)
				return false;
			shaderMacro.Definition = NULL;
			shaderMacros.AddElement(&shaderMacro);
		}
	}
	D3D10_SHADER_MACRO shaderMacro;
	shaderMacro.Name = NULL;
	shaderMacro.Definition = NULL;
	shaderMacros.AddElement(&shaderMacro);
	return true;
}

bool DX11_SHADER::LoadShaderBin(const char *filePath)
{
	FILE *file;
	fopen_s(&file,filePath,"rb");
	if(!file)
		return false;

	char idString[10];
	memset(idString,0,10);
  fread(idString,sizeof(char),9,file);
	if(strcmp(idString,"DEMO_HLSL")!=0)
		return false;

	int version;
	fread(&version,sizeof(int),1,file);
	if(version!=CURRENT_SHADER_VERSION)
		return false;

	fread(uniformBufferMasks,sizeof(int),NUM_UNIFORM_BUFFER_BP,file);
  fread(textureMasks,sizeof(int),NUM_TEXTURE_BP+NUM_STRUCTURED_BUFFER_BP,file);
   
	int numShaders;
	fread(&numShaders,sizeof(int),1,file);
  for(int i=0;i<numShaders;i++)
	{
		int shaderType;
		fread(&shaderType,sizeof(int),1,file);
		int shaderSize;
		fread(&shaderSize,sizeof(int),1,file);
		unsigned char *byteCode = new unsigned char[shaderSize];
		if(!byteCode)
		{
			fclose(file);
			return false;
		}
		fread(byteCode,shaderSize,1,file);
		if(!CreateShaderUnit((shaderTypes)shaderType,byteCode,shaderSize))
		{
			SAFE_DELETE_ARRAY(byteCode);
			fclose(file);
			return false;
		}
		SAFE_DELETE_ARRAY(byteCode);
	}

	fclose(file);
	return true;
}

bool DX11_SHADER::SaveShaderBin(const char *filePath)
{
	FILE *file;
	fopen_s(&file,filePath,"wb");
	if(!file)
		return false;
 
	fwrite("DEMO_HLSL",sizeof(char),9,file);
	int version = CURRENT_SHADER_VERSION;
	fwrite(&version,sizeof(int),1,file);
	fwrite(uniformBufferMasks,sizeof(int),NUM_UNIFORM_BUFFER_BP,file);
	fwrite(textureMasks,sizeof(int),NUM_TEXTURE_BP+NUM_STRUCTURED_BUFFER_BP,file);

  int numShaders = 0;
	for(int i=0;i<NUM_SHADER_TYPES;i++)
	{
		if(byteCodes[i])
			numShaders++;
	}

	fwrite(&numShaders,sizeof(int),1,file);
	for(int i=0;i<NUM_SHADER_TYPES;i++)
	{
		if(byteCodes[i])
		{
			fwrite(&i,sizeof(int),1,file);
			int bufferSize = byteCodes[i]->GetBufferSize();
      fwrite(&bufferSize,sizeof(int),1,file);
			fwrite(byteCodes[i]->GetBufferPointer(),bufferSize,1,file);
		}
	}
	
	fclose(file);
	return true;
}

bool DX11_SHADER::Load(const char *fileName,int permutationMask)
{
	strcpy(name,fileName);
	this->permutationMask = permutationMask;
 
#ifdef USE_SHADER_BIN
	const char *shaderBinPath = NULL;
	char binName[DEMO_MAX_STRING];
	char binPath[DEMO_MAX_FILEPATH];
	if(DEMO::fileManager->GetFileName(fileName,binName))
	{
		sprintf(binPath,"../data/shaders/HLSL/bin/%s_%i.bin",binName,permutationMask);
		shaderBinPath = binPath;
		if(LoadShaderBin(shaderBinPath))
			return true;
	}
#endif

	char filePath[DEMO_MAX_FILEPATH];
	if(!DEMO::fileManager->GetFilePath(fileName,filePath))
		return false;
	std::ifstream file(filePath,std::ios::in);
	if(!file.is_open())
		return false;

	std::string str,token;
	file>>str; 
	while(!file.eof())
	{
		if(str=="Defines")
		{
			LoadDefines(file);
			if(!CreateShaderMacros())
			{
				file.close();
				return false;
			}
		}
		else if(str=="VertexShader")
		{
			if(!LoadShaderUnit(VERTEX_SHADER,file))
			{
				file.close();
				return false;
			}
		}
		else if(str=="GeometryShader")
		{
			if(!LoadShaderUnit(GEOMETRY_SHADER,file))
			{
				file.close();
				return false;
			}
		}
		else if(str=="FragmentShader")
		{
			if(!LoadShaderUnit(FRAGMENT_SHADER,file))
			{
				file.close();
				return false;
			}
		}
		else if(str=="ComputeShader")
		{
			if(!LoadShaderUnit(COMPUTE_SHADER,file))
			{
				file.close();
				return false;
			}
		}
		file>>str;
	} 
	file.close();
	
	shaderMacros.Erase();

#ifdef USE_SHADER_BIN
	if(shaderBinPath)
		SaveShaderBin(shaderBinPath);
#endif

	for(int i=0;i<NUM_SHADER_TYPES;i++)
		SAFE_RELEASE(byteCodes[i]);

	return true;
}

void DX11_SHADER::Bind() const
{
	DEMO::renderer->GetDeviceContext()->VSSetShader(vertexShader,NULL,0);
	DEMO::renderer->GetDeviceContext()->GSSetShader(geometryShader,NULL,0);
	DEMO::renderer->GetDeviceContext()->PSSetShader(fragmentShader,NULL,0);
	DEMO::renderer->GetDeviceContext()->CSSetShader(computeShader,NULL,0);
}

void DX11_SHADER::SetUniformBuffer(const DX11_UNIFORM_BUFFER *uniformBuffer) const
{
	if(!uniformBuffer)
		return;
  for(int i=0;i<NUM_SHADER_TYPES;i++)
	{
		if(uniformBufferMasks[uniformBuffer->GetBindingPoint()] & (1<<i))
			uniformBuffer->Bind((shaderTypes)i); 
	}
}

void DX11_SHADER::SetStructuredBuffer(const DX11_STRUCTURED_BUFFER *structuredBuffer) const
{
	if(!structuredBuffer)
		return;
	for(int i=0;i<NUM_SHADER_TYPES;i++)
	{
		if(textureMasks[structuredBuffer->GetBindingPoint()] & (1<<i))
			structuredBuffer->Bind((shaderTypes)i); 
	}
}

void DX11_SHADER::SetTexture(textureBP bindingPoint,const DX11_TEXTURE *texture) const
{
	if(!texture)
		return;
  for(int i=0;i<NUM_SHADER_TYPES;i++)
	{
		if(textureMasks[bindingPoint] & (1<<i))
			texture->Bind(bindingPoint,(shaderTypes)i);
	}
}





