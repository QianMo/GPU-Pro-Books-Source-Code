/////////////////////////////////////////////////////////////////////////////////////////////
// Copyright 2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <d3d11.h>
#include <assert.h>
#include <d3d11shader.h>
#include <d3dcompiler.h>

// Templated shader factory utilities
namespace ShaderFactoryUtil
{
    // Reads file from disk. Doesn't handle memory de-allocation.
    DWORD ReadFileFromDisk(LPCTSTR pFile, void** buffer);

    template <typename T> LPCSTR GetShaderProfileString();
    template <typename T> T* CreateShader(ID3D11Device* d3dDevice, const void* shaderBytecode, size_t bytecodeLength);
    template <typename T> T* CreateShader(ID3D11Device* d3dDevice, LPCTSTR objFile, ID3D11ShaderReflection** reflector);

    // Vertex shader
    template <> inline LPCSTR GetShaderProfileString<ID3D11VertexShader>() { return "vs_5_0"; }
    template <> inline 
    ID3D11VertexShader* CreateShader<ID3D11VertexShader>(ID3D11Device* d3dDevice, const void* shaderBytecode, size_t bytecodeLength)
    {
        ID3D11VertexShader *shader = 0;
        HRESULT hr = d3dDevice->CreateVertexShader(shaderBytecode, bytecodeLength, 0, &shader);
        if (FAILED(hr)) {
            // This shouldn't produce errors given proper bytecode, so a simple assert is fine
            assert(false);
        }
        return shader;
    }

    // Geometry shader
    template <> inline LPCSTR GetShaderProfileString<ID3D11GeometryShader>() { return "gs_5_0"; }
    template <> inline 
    ID3D11GeometryShader* CreateShader<ID3D11GeometryShader>(ID3D11Device* d3dDevice, const void* shaderBytecode, size_t bytecodeLength)
    {
        ID3D11GeometryShader *shader = 0;
        HRESULT hr = d3dDevice->CreateGeometryShader(shaderBytecode, bytecodeLength, 0, &shader);
        if (FAILED(hr)) {
            // This shouldn't produce errors given proper bytecode, so a simple assert is fine
            assert(false);
        }
        return shader;
    }

    // Pixel shader
    template <> inline LPCSTR GetShaderProfileString<ID3D11PixelShader>() { return "ps_5_0"; }
    template <> inline 
    ID3D11PixelShader* CreateShader<ID3D11PixelShader>(ID3D11Device* d3dDevice, const void* shaderBytecode, size_t bytecodeLength)
    {
        ID3D11PixelShader *shader = 0;
        HRESULT hr = d3dDevice->CreatePixelShader(shaderBytecode, bytecodeLength, 0, &shader);
        if (FAILED(hr)) {
            // This shouldn't produce errors given proper bytecode, so a simple assert is fine
            assert(false);
        }
        return shader;
    }

    template <> inline 
    ID3D11PixelShader * CreateShader<ID3D11PixelShader>(ID3D11Device* d3dDevice, LPCTSTR objFile, ID3D11ShaderReflection** reflector)
    {
        ID3D11PixelShader *shader = 0;
        void* fileBuffer;
        DWORD fileSize = ReadFileFromDisk(objFile, &fileBuffer);

        // Create shader and reflector
        HRESULT hr;
        hr = d3dDevice->CreatePixelShader(fileBuffer, fileSize, 0, &shader);
        assert(SUCCEEDED(hr));

        if (reflector) {
            D3DReflect(fileBuffer, fileSize, IID_ID3D11ShaderReflection, (void**) reflector);
        }

        free(fileBuffer);
        return shader;
    }



    // Compute shader
    template <> inline LPCSTR GetShaderProfileString<ID3D11ComputeShader>() { return "cs_5_0"; }
    template <> inline 
    ID3D11ComputeShader* CreateShader<ID3D11ComputeShader>(ID3D11Device* d3dDevice, const void* shaderBytecode, size_t bytecodeLength)
    {
        ID3D11ComputeShader *shader = 0;
        HRESULT hr = d3dDevice->CreateComputeShader(shaderBytecode, bytecodeLength, 0, &shader);
        if (FAILED(hr)) {
            // This shouldn't produce errors given proper bytecode, so a simple assert is fine
            assert(false);
        }
        return shader;
    }

    template <> inline 
    ID3D11ComputeShader* CreateShader<ID3D11ComputeShader>(ID3D11Device* d3dDevice, LPCTSTR objFile, ID3D11ShaderReflection** reflector)
    {
        ID3D11ComputeShader *shader = 0;
        void* fileBuffer;
        DWORD fileSize = ReadFileFromDisk(objFile, &fileBuffer);

        // Create shader and reflector
        HRESULT hr;
        hr = d3dDevice->CreateComputeShader(fileBuffer, fileSize, 0, &shader);
        assert(SUCCEEDED(hr));

        if (reflector) {
            D3DReflect(fileBuffer, fileSize, IID_ID3D11ShaderReflection, (void**) reflector);
        }

        free(fileBuffer);
        return shader;
    }
}

// Templated (on shader type) shader wrapper to wrap basic functionality
// TODO: Support optional lazy compile
template <typename T>
class Shader
{
public:
    Shader(ID3D11Device* d3dDevice, LPCTSTR srcFile, LPCSTR functionName, CONST D3D10_SHADER_MACRO *defines = 0)
    {
        // TODO: Support profile selection from the application? Probably not necessary as we don't
        // support down-level hardware at the moment anyways.
        LPCSTR profile = ShaderFactoryUtil::GetShaderProfileString<T>();

        UINT shaderFlags = D3D10_SHADER_ENABLE_STRICTNESS | D3D10_SHADER_PACK_MATRIX_ROW_MAJOR;

        ID3D10Blob *bytecode = 0;
        HRESULT hr = D3DCompileFromFile(srcFile, defines, D3D_COMPILE_STANDARD_FILE_INCLUDE, functionName, profile, shaderFlags, 0, &bytecode, 0);
        
        if (FAILED(hr)) {
            // TODO: Define exception type and improve this error string, but the ODS will do for now
            throw std::runtime_error("Error compiling shader");
        }

        mShader = ShaderFactoryUtil::CreateShader<T>(d3dDevice, bytecode->GetBufferPointer(), bytecode->GetBufferSize());

        bytecode->Release();
    }


    Shader(ID3D11Device* d3dDevice, ID3D11ShaderReflection** reflector, LPCTSTR objFile)
    {
        mShader = ShaderFactoryUtil::CreateShader<T>(d3dDevice, objFile, reflector);
    }

    ~Shader() { mShader->Release(); }

    T* GetShader() { return mShader; }

private:
    // Not implemented
    Shader(const Shader&);
    Shader& operator=(const Shader&);

    T* mShader;
};


typedef Shader<ID3D11VertexShader> VertexShader;
typedef Shader<ID3D11GeometryShader> GeometryShader;
typedef Shader<ID3D11PixelShader> PixelShader;
typedef Shader<ID3D11ComputeShader> ComputeShader;