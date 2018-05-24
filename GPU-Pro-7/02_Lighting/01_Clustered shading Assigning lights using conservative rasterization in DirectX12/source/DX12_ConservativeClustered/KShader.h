#pragma once
#include <D3D12.h>
#include <d3dcompiler.h>
#include <d3d12shader.h>
#include <vector>
#include "Types.h"

class KShader
{
public:
	KShader(LPCWSTR file_path, const char* entry_point, const char* target, const D3D_SHADER_MACRO* macro_list = nullptr);
	~KShader();

	LPVOID GetBufferPointer()	{ return m_Blob->GetBufferPointer(); }
	SIZE_T GetBufferSize()		{ return m_Blob->GetBufferSize(); }

	D3D12_INPUT_LAYOUT_DESC GetInputLayout();

private:

	//Shader byte code
	ID3DBlob* m_Blob;
	ID3D12ShaderReflection* m_Reflection;

	std::vector<D3D12_INPUT_ELEMENT_DESC> m_InputElementDesc;

	void PrintShaderInfo(ID3D12ShaderReflection* reflection);
};