// Copyright 2010 Intel Corporation
// All Rights Reserved
//
// Permission is granted to use, copy, distribute and prepare derivative works of this
// software for any purpose and without fee, provided, that the above copyright notice
// and this statement appear in all copies.  Intel makes no representations about the
// suitability of this software for any purpose.  THIS SOFTWARE IS PROVIDED "AS IS."
// INTEL SPECIFICALLY DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, AND ALL LIABILITY,
// INCLUDING CONSEQUENTIAL AND OTHER INDIRECT DAMAGES, FOR THE USE OF THIS SOFTWARE,
// INCLUDING LIABILITY FOR INFRINGEMENT OF ANY PROPRIETARY RIGHTS, AND INCLUDING THE
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  Intel does not
// assume any responsibility for any errors which may appear in this software nor any
// responsibility to update it.

#include "ShaderUtils.h"

#include <algorithm>
#include <limits>

namespace ShaderUtils {

const unsigned int kFlags = 0;//D3D10_SHADER_SKIP_OPTIMIZATION;

void CreateVertexShader(ID3D11Device *device, 
                        LPCTSTR fileName, 
                        LPCSTR  functionName, 
                        LPCSTR  profileName, 
                        CONST D3D10_SHADER_MACRO *defines, 
                        UINT shaderFlags, 
                        ID3D11VertexShader **shader,
                        ID3D11ShaderReflection **reflector)
{      
    HRESULT hr;
    // Vertex shader
    ID3D10Blob *vertexShaderBlob = NULL;
    ID3D10Blob *errorsBlob = NULL;
    hr = D3DX11CompileFromFile(fileName, defines, 0,
                               functionName ,profileName,
                               shaderFlags | kFlags,
                               0, 0, 
                               &vertexShaderBlob, 
                               &errorsBlob, 
                               NULL);
    if (errorsBlob) {
        const char *errorMsg =
            reinterpret_cast<const char*>(errorsBlob->GetBufferPointer());
        OutputDebugStringA(errorMsg);
    }
    // Do something better here on shader compile failure...
    assert(SUCCEEDED(hr));

    hr = device->CreateVertexShader(vertexShaderBlob->GetBufferPointer(),
                                    vertexShaderBlob->GetBufferSize(),
                                    0,
                                    shader);
    assert(SUCCEEDED(hr));

    if (reflector) {
        D3DReflect(vertexShaderBlob->GetBufferPointer(),
                   vertexShaderBlob->GetBufferSize(), 
                   IID_ID3D11ShaderReflection, (void**) reflector);
    }

    vertexShaderBlob->Release();
}


// Reads file from disk. Doesn't handle memory de-allocation.
DWORD ReadFileFromDisk(LPCTSTR pFile,
                       void** buffer)
{
    // Open compiled shader file
    HANDLE fileHandle = CreateFile(pFile, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
    assert(fileHandle != INVALID_HANDLE_VALUE);
       
    DWORD fileSize = GetFileSize(fileHandle, NULL);
    assert(fileSize != INVALID_FILE_SIZE);

    // We need enough mem to load the file from disk
    *buffer = malloc(fileSize);
    assert(*buffer != NULL);

    // Read file from disk
    BOOL bhr;
    DWORD wmWritten;
    bhr = ReadFile(fileHandle, (LPVOID)(*buffer), fileSize, &wmWritten, NULL);
    assert(bhr != 0);

    // Close file
    CloseHandle(fileHandle);

    return fileSize;
}

void CreateVertexShaderFromCompiledObj(ID3D11Device *device, 
                                       LPCTSTR pFile, 
                                       ID3D11VertexShader** shader,
                                       ID3D11ShaderReflection** reflector)
{
    void* fileBuffer;
    DWORD fileSize = ReadFileFromDisk(pFile, &fileBuffer);

    // Create shader and reflector
    HRESULT hr;
    hr = device->CreateVertexShader(fileBuffer, fileSize, 0, shader);
    assert(SUCCEEDED(hr));

    if (reflector) {
        D3DReflect(fileBuffer, fileSize, IID_ID3D11ShaderReflection, (void**) reflector);
    }
    
    free(fileBuffer);
}

void CreateGeometryShaderFromCompiledObj(ID3D11Device *device, 
                                         LPCTSTR pFile, 
                                         ID3D11GeometryShader** shader,
                                         ID3D11ShaderReflection** reflector)
{
    void* fileBuffer;
    DWORD fileSize = ReadFileFromDisk(pFile, &fileBuffer);

    // Create shader and reflector
    HRESULT hr;
    hr = device->CreateGeometryShader(fileBuffer, fileSize, 0, shader);
    assert(SUCCEEDED(hr));

    if (reflector) {
        D3DReflect(fileBuffer, fileSize, IID_ID3D11ShaderReflection, (void**) reflector);
    }
    
    free(fileBuffer);
}

void CreatePixelShaderFromCompiledObj(ID3D11Device *device, 
                                      LPCTSTR pFile, 
                                      ID3D11PixelShader** shader,
                                      ID3D11ShaderReflection** reflector)
{
    void* fileBuffer;
    DWORD fileSize = ReadFileFromDisk(pFile, &fileBuffer);

    // Create shader and reflector
    HRESULT hr;
    hr = device->CreatePixelShader(fileBuffer, fileSize, 0, shader);
    assert(SUCCEEDED(hr));

    if (reflector) {
        D3DReflect(fileBuffer, fileSize, IID_ID3D11ShaderReflection, (void**) reflector);
    }
    
    free(fileBuffer);
}

void CreateComputeShaderFromCompiledObj(ID3D11Device *device, 
                                      LPCTSTR pFile, 
                                      ID3D11ComputeShader** shader,
                                      ID3D11ShaderReflection** reflector)
{
    void* fileBuffer;
    DWORD fileSize = ReadFileFromDisk(pFile, &fileBuffer);

    // Create shader and reflector
    HRESULT hr;
    hr = device->CreateComputeShader(fileBuffer, fileSize, 0, shader);
    assert(SUCCEEDED(hr));

    if (reflector) {
        D3DReflect(fileBuffer, fileSize, IID_ID3D11ShaderReflection, (void**) reflector);
    }
    
    free(fileBuffer);
}


void CreatePixelShader(ID3D11Device *device, 
                       LPCTSTR fileName, 
                       LPCSTR  functionName, 
                       LPCSTR  profileName, 
                       CONST D3D10_SHADER_MACRO *defines, 
                       UINT shaderFlags, 
                       ID3D11PixelShader **shader,
                       ID3D11ShaderReflection **reflector)
{
    HRESULT hr;
    // Pixel shader
    ID3D10Blob *pixelShaderBlob = NULL;
    ID3D10Blob *errorsBlob = NULL;
    hr = D3DX11CompileFromFile(fileName, defines, 0,
                               functionName ,profileName,
                               shaderFlags | kFlags,
                               0, 0, 
                               &pixelShaderBlob,
                               &errorsBlob, 
                               NULL);
    if (errorsBlob) {
        const char *errorMsg =
            reinterpret_cast<const char*>(errorsBlob->GetBufferPointer());
        OutputDebugStringA(errorMsg);
    }
    // Do something better here on shader compile failure...
    assert(SUCCEEDED(hr));

    hr = device->CreatePixelShader(pixelShaderBlob->GetBufferPointer(),
                                   pixelShaderBlob->GetBufferSize(),
                                   0,
                                   shader);
    assert(SUCCEEDED(hr));

    if (reflector) {
        D3DReflect(pixelShaderBlob->GetBufferPointer(),
                   pixelShaderBlob->GetBufferSize(), 
                   IID_ID3D11ShaderReflection, (void**) reflector);
    }

    pixelShaderBlob->Release();
}


void CreateComputeShader(ID3D11Device *device, 
                         LPCTSTR fileName, 
                         LPCSTR  functionName, 
                         LPCSTR  profileName, 
                         CONST D3D10_SHADER_MACRO *defines, 
                         UINT shaderFlags, 
                         ID3D11ComputeShader **shader,
                         ID3D11ShaderReflection **reflector)
{
    HRESULT hr;
    // Pixel shader
    ID3D10Blob *computeShaderBlob = NULL;
    ID3D10Blob *errorsBlob = NULL;
    hr = D3DX11CompileFromFile(fileName, defines, 0,
                               functionName ,profileName,
                               shaderFlags | kFlags,
                               0, 0, 
                               &computeShaderBlob, 
                               &errorsBlob, 
                               NULL);
    if (errorsBlob) {
        const char *errorMsg =
            reinterpret_cast<const char*>(errorsBlob->GetBufferPointer());
        OutputDebugStringA(errorMsg);
    }
    // Do something better here on shader compile failure...
    assert(SUCCEEDED(hr));

    hr = device->CreateComputeShader(computeShaderBlob->GetBufferPointer(),
                                     computeShaderBlob->GetBufferSize(),
                                     0,
                                     shader);
    assert(SUCCEEDED(hr));

    if (reflector) {
        D3DReflect(computeShaderBlob->GetBufferPointer(),
                   computeShaderBlob->GetBufferSize(), 
                   IID_ID3D11ShaderReflection, (void**) reflector);
    }

    computeShaderBlob->Release();
}

void VSSetConstantBuffers(ID3D11DeviceContext* d3dDeviceContext,
                          ID3D11ShaderReflection* shaderReflection,
                          const char *name,
                          UINT count,
                          ID3D11Buffer** buffers)
{
    UINT bindIndex;
    if (SUCCEEDED(GetBindIndex(shaderReflection, name, &bindIndex))) {
        d3dDeviceContext->VSSetConstantBuffers(bindIndex,
                                               count,
                                               buffers);
    }
}

void GSSetConstantBuffers(ID3D11DeviceContext* d3dDeviceContext,
                          ID3D11ShaderReflection* shaderReflection,
                          const char *name,
                          UINT count,
                          ID3D11Buffer** buffers)
{
    UINT bindIndex;
    if (SUCCEEDED(GetBindIndex(shaderReflection, name, &bindIndex))) {
        d3dDeviceContext->GSSetConstantBuffers(bindIndex,
                                               count,
                                               buffers);
    }
}

void PSSetConstantBuffers(ID3D11DeviceContext* d3dDeviceContext,
                          ID3D11ShaderReflection* shaderReflection,
                          const char *name,
                          UINT count,
                          ID3D11Buffer** buffers)
{
    UINT bindIndex;
    if (SUCCEEDED(GetBindIndex(shaderReflection, name, &bindIndex))) {
        d3dDeviceContext->PSSetConstantBuffers(bindIndex,
                                               count,
                                               buffers);
    }
}

void CSSetConstantBuffers(ID3D11DeviceContext* d3dDeviceContext,
                          ID3D11ShaderReflection* shaderReflection,
                          const char *name,
                          UINT count,
                          ID3D11Buffer** buffers)
{
    UINT bindIndex;
    if (SUCCEEDED(GetBindIndex(shaderReflection, name, &bindIndex))) {
        d3dDeviceContext->CSSetConstantBuffers(bindIndex,
                                               count,
                                               buffers);
    }
}

void VSSetShaderResources(ID3D11DeviceContext* d3dDeviceContext,
                          ID3D11ShaderReflection* shaderReflection,
                          const char *name,
                          UINT count,
                          ID3D11ShaderResourceView** SRVs)
{
    UINT bindIndex;
    if (SUCCEEDED(GetBindIndex(shaderReflection, name, &bindIndex))) {
        d3dDeviceContext->VSSetShaderResources(bindIndex,
                                               count,
                                               SRVs);
    }
}

void GSSetShaderResources(ID3D11DeviceContext* d3dDeviceContext,
                          ID3D11ShaderReflection* shaderReflection,
                          const char *name,
                          UINT count,
                          ID3D11ShaderResourceView** SRVs)
{
    UINT bindIndex;
    if (SUCCEEDED(GetBindIndex(shaderReflection, name, &bindIndex))) {
        d3dDeviceContext->GSSetShaderResources(bindIndex,
                                               count,
                                               SRVs);
    }
}

void PSSetShaderResources(ID3D11DeviceContext* d3dDeviceContext,
                          ID3D11ShaderReflection* shaderReflection,
                          const char *name,
                          UINT count,
                          ID3D11ShaderResourceView** SRVs)
{
    UINT bindIndex;
    if (SUCCEEDED(GetBindIndex(shaderReflection, name, &bindIndex))) {
        d3dDeviceContext->PSSetShaderResources(bindIndex,
                                               count,
                                               SRVs);
    }
}

void CSSetShaderResources(ID3D11DeviceContext* d3dDeviceContext,
                          ID3D11ShaderReflection* shaderReflection,
                          const char *name,
                          UINT count,
                          ID3D11ShaderResourceView** SRVs)
{
    UINT bindIndex;
    if (SUCCEEDED(GetBindIndex(shaderReflection, name, &bindIndex))) {
        d3dDeviceContext->CSSetShaderResources(bindIndex,
                                               count,
                                               SRVs);
    }
}

void VSSetSamplers(ID3D11DeviceContext* d3dDeviceContext,
                   ID3D11ShaderReflection* shaderReflection,
                   const char *name,
                   UINT count,
                   ID3D11SamplerState** samplers)
{
    UINT bindIndex;
    if (SUCCEEDED(GetBindIndex(shaderReflection, name, &bindIndex))) {
        d3dDeviceContext->VSSetSamplers(bindIndex, count, samplers);
    }
}

void GSSetSamplers(ID3D11DeviceContext* d3dDeviceContext,
                   ID3D11ShaderReflection* shaderReflection,
                   const char *name,
                   UINT count,
                   ID3D11SamplerState** samplers)
{
    UINT bindIndex;
    if (SUCCEEDED(GetBindIndex(shaderReflection, name, &bindIndex))) {
        d3dDeviceContext->GSSetSamplers(bindIndex, count, samplers);
    }
}

void PSSetSamplers(ID3D11DeviceContext* d3dDeviceContext,
                   ID3D11ShaderReflection* shaderReflection,
                   const char *name,
                   UINT count,
                   ID3D11SamplerState** samplers)
{
    UINT bindIndex;
    if (SUCCEEDED(GetBindIndex(shaderReflection, name, &bindIndex))) {
        d3dDeviceContext->PSSetSamplers(bindIndex, count, samplers);
    }
}

void CSSetSamplers(ID3D11DeviceContext* d3dDeviceContext,
                   ID3D11ShaderReflection* shaderReflection,
                   const char *name,
                   UINT count,
                   ID3D11SamplerState** samplers)
{
    UINT bindIndex;
    if (SUCCEEDED(GetBindIndex(shaderReflection, name, &bindIndex))) {
        d3dDeviceContext->CSSetSamplers(bindIndex, count, samplers);
    }
}

void CSSetUnorderedAccessViews(ID3D11DeviceContext* d3dDeviceContext,
                               ID3D11ShaderReflection* shaderReflection,
                               const char *name,
                               UINT count,
                               ID3D11UnorderedAccessView **UAVs,
                               const UINT *UAVInitialCounts)
{
    UINT bindIndex;
    if (SUCCEEDED(GetBindIndex(shaderReflection, name, &bindIndex))) {
        d3dDeviceContext->CSSetUnorderedAccessViews(bindIndex, 
                                                    count, 
                                                    UAVs, 
                                                    UAVInitialCounts);
    }
}

HRESULT GetBindIndex(ID3D11ShaderReflection* shaderReflection, 
                     const char *name,
                     UINT *bindIndex)
{
    HRESULT hr;
    D3D11_SHADER_INPUT_BIND_DESC inputBindDesc;
    hr = shaderReflection->GetResourceBindingDescByName(name,
                                                        &inputBindDesc);
    if (SUCCEEDED(hr)) {
        *bindIndex = inputBindDesc.BindPoint;
    } else {
        char buffer[4096];
        sprintf_s(buffer, "Could not find shader parameter %s\n", name);
        OutputDebugStringA(buffer);
    }
    return hr;
}

UINT GetStartBindIndex(ID3D11ShaderReflection* shaderReflection,
                       const char *paramUAVs[],
                       const UINT numUAVs)
{
    UINT minIndex = std::numeric_limits<unsigned int>::max();
    for (UINT i = 0; i < numUAVs; ++i) {
        UINT index = 0;
        HRESULT hr;
        hr = GetBindIndex(shaderReflection, paramUAVs[i], &index);
        assert(SUCCEEDED(hr));
        minIndex = std::min(minIndex, index);
    }
    return minIndex;
}

}
