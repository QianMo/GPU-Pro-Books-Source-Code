#pragma once

#include <d3d12.h>
#include <d3d12shader.h>

namespace NGraphics
{
    class CShader
    {
    private:
        ID3DBlob* m_Code;
        ID3D12ShaderReflection* m_Reflection;

        UINT m_InputElementDescCount;
        D3D12_INPUT_ELEMENT_DESC* m_InputElementDescArray;

    public:
        CShader( LPCWSTR filepath, const char* entry_point, const char* target, const D3D_SHADER_MACRO* defines = nullptr );
        ~CShader();

        D3D12_SHADER_BYTECODE GetShaderBytecode();
        D3D12_INPUT_LAYOUT_DESC GetInputLayout();
    };
}