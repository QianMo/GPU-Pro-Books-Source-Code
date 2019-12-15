#ifndef DX12_SHADER_H
#define DX12_SHADER_H

#include <list.h>
#include <render_states.h>

// DX12_Shader
//
// Loaded from a simple text-file (".sdr"), that references the actual shader source files.
class DX12_Shader
{
public:
	friend class ShaderInclude;
	
	DX12_Shader():
    permutationMask(0),
    shaderUnitMask(0)
	{
    name[0] = 0;
	}

	bool Load(const char *fileName, UINT permutationMask=0);

  D3D12_SHADER_BYTECODE GetByteCode(shaderTypes shaderType);

	UINT GetPermutationMask() const
	{
		return permutationMask;
	}

	const char* GetName() const
	{
		return name;
	}

private:
  void LoadDefines(std::ifstream &file);

  bool CreateShaderMacros();

	bool LoadShaderUnit(shaderTypes shaderType, std::ifstream &file);

  List<char[DEMO_MAX_STRING]> defineStrings;
  UINT permutationMask;
  UINT shaderUnitMask;
  char name[DEMO_MAX_FILENAME];

  ComPtr<ID3DBlob> byteCodes[NUM_SHADER_TYPES];
	List<D3D_SHADER_MACRO> shaderMacros;

};

#endif
