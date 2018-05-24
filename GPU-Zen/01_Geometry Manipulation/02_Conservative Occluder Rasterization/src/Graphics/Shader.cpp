#include "Shader.h"
#include "GraphicsDefines.h"

#include <d3dcompiler.h>
#include <stdio.h>
#include <assert.h>

namespace NGraphics
{
    CShader::CShader( LPCWSTR filepath, const char* entry_point, const char* target, const D3D_SHADER_MACRO* defines ) :
        m_Code( nullptr ),
        m_Reflection( nullptr ),
        m_InputElementDescCount( 0 ),
        m_InputElementDescArray( nullptr )
    {
        UINT compile_flags = D3DCOMPILE_OPTIMIZATION_LEVEL3;
    #ifdef _DEBUG
        compile_flags = D3DCOMPILE_DEBUG;
    #endif

        ID3DBlob* error = nullptr;
        HRESULT hr = D3DCompileFromFile( filepath, defines, D3D_COMPILE_STANDARD_FILE_INCLUDE, entry_point, target, compile_flags, 0, &m_Code, &error );
        if ( FAILED( hr ) )
        {
            if ( hr == HRESULT_FROM_WIN32( ERROR_FILE_NOT_FOUND ) )
                printf( "File %ws not found.\n", filepath );
            else if ( error != nullptr )
                printf( "Error compiling file %ws: %s\n", filepath, ( char* )error->GetBufferPointer() );
            else
                printf( "Error compiling file %ws.\n", filepath );
            throw;
        }
        if ( error != nullptr )
        {
            error->Release();
        }
        
        D3DReflect( m_Code->GetBufferPointer(), m_Code->GetBufferSize(), IID_PPV_ARGS( &m_Reflection ) );

        D3D12_SHADER_DESC shader_desc;
        m_Reflection->GetDesc( &shader_desc );

        m_InputElementDescCount = shader_desc.InputParameters;
        m_InputElementDescArray = new D3D12_INPUT_ELEMENT_DESC[ m_InputElementDescCount ];
        for ( UINT i = 0; i < shader_desc.InputParameters; ++i )
        {
            D3D12_SIGNATURE_PARAMETER_DESC parameter_desc;
            m_Reflection->GetInputParameterDesc( i, &parameter_desc );

            ZeroMemory( &m_InputElementDescArray[ i ], sizeof( m_InputElementDescArray[ i ] ) );
            m_InputElementDescArray[ i ].SemanticName = parameter_desc.SemanticName;
            m_InputElementDescArray[ i ].SemanticIndex = parameter_desc.SemanticIndex;
            m_InputElementDescArray[ i ].AlignedByteOffset = D3D12_APPEND_ALIGNED_ELEMENT;

            if ( parameter_desc.Mask == 1 )
            {
                if ( parameter_desc.ComponentType == D3D_REGISTER_COMPONENT_UINT32 )
                    m_InputElementDescArray[ i ].Format = DXGI_FORMAT_R32_UINT;
                else if ( parameter_desc.ComponentType == D3D_REGISTER_COMPONENT_SINT32 )
                    m_InputElementDescArray[ i ].Format = DXGI_FORMAT_R32_SINT;
                else if ( parameter_desc.ComponentType == D3D_REGISTER_COMPONENT_FLOAT32 )
                    m_InputElementDescArray[ i ].Format = DXGI_FORMAT_R32_FLOAT;
            }
            else if ( parameter_desc.Mask <= 3 )
            {
                if ( parameter_desc.ComponentType == D3D_REGISTER_COMPONENT_UINT32 )
                    m_InputElementDescArray[ i ].Format = DXGI_FORMAT_R32G32_UINT;
                else if ( parameter_desc.ComponentType == D3D_REGISTER_COMPONENT_SINT32 )
                    m_InputElementDescArray[ i ].Format = DXGI_FORMAT_R32G32_SINT;
                else if ( parameter_desc.ComponentType == D3D_REGISTER_COMPONENT_FLOAT32 )
                    m_InputElementDescArray[ i ].Format = DXGI_FORMAT_R32G32_FLOAT;
            }
            else if ( parameter_desc.Mask <= 7 )
            {
                if ( parameter_desc.ComponentType == D3D_REGISTER_COMPONENT_UINT32 )
                    m_InputElementDescArray[ i ].Format = DXGI_FORMAT_R32G32B32_UINT;
                else if ( parameter_desc.ComponentType == D3D_REGISTER_COMPONENT_SINT32 )
                    m_InputElementDescArray[ i ].Format = DXGI_FORMAT_R32G32B32_SINT;
                else if ( parameter_desc.ComponentType == D3D_REGISTER_COMPONENT_FLOAT32 )
                    m_InputElementDescArray[ i ].Format = DXGI_FORMAT_R32G32B32_FLOAT;
            }
            else if ( parameter_desc.Mask <= 15 )
            {
                if ( parameter_desc.ComponentType == D3D_REGISTER_COMPONENT_UINT32 )
                    m_InputElementDescArray[ i ].Format = DXGI_FORMAT_R32G32B32A32_UINT;
                else if ( parameter_desc.ComponentType == D3D_REGISTER_COMPONENT_SINT32 )
                    m_InputElementDescArray[ i ].Format = DXGI_FORMAT_R32G32B32A32_SINT;
                else if ( parameter_desc.ComponentType == D3D_REGISTER_COMPONENT_FLOAT32 )
                    m_InputElementDescArray[ i ].Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
            }
        }
    }

    CShader::~CShader()
    {
        m_InputElementDescCount = 0;

        SAFE_DELETE_ARRAY( m_InputElementDescArray );
        SAFE_RELEASE( m_Reflection );
        SAFE_RELEASE( m_Code );
    }

    D3D12_SHADER_BYTECODE CShader::GetShaderBytecode()
    {
        return { m_Code->GetBufferPointer(), static_cast< UINT >( m_Code->GetBufferSize() ) };
    }

    D3D12_INPUT_LAYOUT_DESC CShader::GetInputLayout()
    {
        return { m_InputElementDescArray, m_InputElementDescCount };
    }
}