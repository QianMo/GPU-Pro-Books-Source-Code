#ifndef DX11_SHADER_H
#define DX11_SHADER_H

#include <LIST.h>
#include <render_states.h>

#define CURRENT_SHADER_VERSION 1 // current version of binary pre-compiled shaders

class DX11_UNIFORM_BUFFER; 
class DX11_STRUCTURED_BUFFER;
class DX11_TEXTURE;

// DX11_SHADER
//   Loaded from a simple text-file (".sdr"), that references the actual shader source files.
//   To avoid long loading times, per default pre-compiled shaders are used.
class DX11_SHADER
{
public:
	friend class SHADER_INCLUDE;
	
	DX11_SHADER()
	{
		permutationMask = 0;
		vertexShader = NULL;
		geometryShader = NULL;
		fragmentShader = NULL;
		computeShader = NULL;
		shaderMacros.Erase();
		for(int i=0;i<NUM_SHADER_TYPES;i++)
		  byteCodes[i] = NULL;
		for(int i=0;i<NUM_UNIFORM_BUFFER_BP;i++)
			uniformBufferMasks[i] = 0;
		for(int i=0;i<(NUM_TEXTURE_BP+NUM_STRUCTURED_BUFFER_BP);i++)
			textureMasks[i] = 0;
	}

	~DX11_SHADER()
	{
		Release();
	}

	void Release();

	bool Load(const char *fileName,int permutationMask=0);

	void Bind() const;

	void SetUniformBuffer(const DX11_UNIFORM_BUFFER *uniformBuffer) const;

	void SetStructuredBuffer(const DX11_STRUCTURED_BUFFER *structuredBuffer) const;

	void SetTexture(textureBP bindingPoint,const DX11_TEXTURE *texture) const;

	int GetPermutationMask() const
	{
		return permutationMask;
	}

	const char* GetName() const
	{
		return name;
	}

private:
  void LoadDefines(std::ifstream &file);

	static bool ReadShaderFile(const char *fileName,unsigned char **data,unsigned int *dataSize);

	bool CreateShaderUnit(shaderTypes shaderType,void *byteCode,int shaderSize);
	
	bool InitShaderUnit(shaderTypes shaderType,const char *fileName);

	bool LoadShaderUnit(shaderTypes shaderType,std::ifstream &file);

	bool CreateShaderMacros();

	void ParseShaderString(shaderTypes shaderType,const char *shaderString);

	bool LoadShaderBin(const char *filePath);

	bool SaveShaderBin(const char *filePath);

	LIST<char[DEMO_MAX_STRING]> defineStrings;
	int permutationMask;
	char name[DEMO_MAX_FILENAME];

	ID3D11VertexShader *vertexShader;
	ID3D11GeometryShader *geometryShader; 
	ID3D11PixelShader *fragmentShader; 
	ID3D11ComputeShader *computeShader;
	LIST<D3D10_SHADER_MACRO> shaderMacros;
	ID3DBlob *byteCodes[NUM_SHADER_TYPES];

	int uniformBufferMasks[NUM_UNIFORM_BUFFER_BP];
	int textureMasks[NUM_TEXTURE_BP+NUM_STRUCTURED_BUFFER_BP];

	static const char *shaderModels[NUM_SHADER_TYPES];
	static const char *uniformBufferRegisterNames[NUM_UNIFORM_BUFFER_BP];
	static const char *textureRegisterNames[NUM_TEXTURE_BP+NUM_STRUCTURED_BUFFER_BP]; 

};

#endif
