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

#ifndef H_SHADER_UTILS
#define H_SHADER_UTILS

#include "DXUT.h"

namespace ShaderUtils
{
    // The app is still responsible for releasing shader objects
    void CreateVertexShader(ID3D11Device *d3dDevice, 
                            LPCTSTR pFile, 
                            LPCSTR  functionName, 
                            LPCSTR profile, 
                            CONST D3D10_SHADER_MACRO *pDefines, 
                            UINT shaderFlags, 
                            ID3D11VertexShader** shader,
                            ID3D11ShaderReflection** reflector);

    void CreatePixelShader(ID3D11Device *d3dDevice, 
                           LPCTSTR pFile, 
                           LPCSTR  functionName, 
                           LPCSTR profile, 
                           CONST D3D10_SHADER_MACRO *pDefines, 
                           UINT shaderFlags, 
                           ID3D11PixelShader** shader,
                           ID3D11ShaderReflection** reflector);

    void CreateComputeShader(ID3D11Device *d3dDevice, 
                             LPCTSTR pFile, 
                             LPCSTR  functionName, 
                             LPCSTR profile, 
                             CONST D3D10_SHADER_MACRO *pDefines, 
                             UINT shaderFlags, 
                             ID3D11ComputeShader** shader,
                             ID3D11ShaderReflection** reflector);

    DWORD ReadFileFromDisk(LPCTSTR pFile,
                           void** buffer,
                           ID3D11VertexShader** shader,
                           ID3D11ShaderReflection** reflector);


    void CreateVertexShaderFromCompiledObj(ID3D11Device *d3dDevice, 
                                           LPCTSTR pFile, 
                                           ID3D11VertexShader** shader,
                                           ID3D11ShaderReflection** reflector);

    void CreateGeometryShaderFromCompiledObj(ID3D11Device *d3dDevice, 
                                             LPCTSTR pFile, 
                                             ID3D11GeometryShader** shader,
                                             ID3D11ShaderReflection** reflector);

    void CreatePixelShaderFromCompiledObj(ID3D11Device *d3dDevice, 
                                          LPCTSTR pFile, 
                                          ID3D11PixelShader** shader,
                                          ID3D11ShaderReflection** reflector);

    void CreateComputeShaderFromCompiledObj(ID3D11Device *d3dDevice, 
                                            LPCTSTR pFile, 
                                            ID3D11ComputeShader** shader,
                                            ID3D11ShaderReflection** reflector);

    void VSSetConstantBuffers(ID3D11DeviceContext* d3dDeviceContext,
                              ID3D11ShaderReflection* shaderReflection,
                              const char *name,
                              UINT count,
                              ID3D11Buffer** buffers);
    void GSSetConstantBuffers(ID3D11DeviceContext* d3dDeviceContext,
                              ID3D11ShaderReflection* shaderReflection,
                              const char *name,
                              UINT count,
                              ID3D11Buffer** buffers);
    void PSSetConstantBuffers(ID3D11DeviceContext* d3dDeviceContext,
                              ID3D11ShaderReflection* shaderReflection,
                              const char *name,
                              UINT count,
                              ID3D11Buffer** buffers);
    void CSSetConstantBuffers(ID3D11DeviceContext* d3dDeviceContext,
                              ID3D11ShaderReflection* shaderReflection,
                              const char *name,
                              UINT count,
                              ID3D11Buffer** buffers);

    void VSSetShaderResources(ID3D11DeviceContext* d3dDeviceContext,
                              ID3D11ShaderReflection* shaderReflection,
                              const char *name,
                              UINT count,
                              ID3D11ShaderResourceView** SRVs);
    void GSSetShaderResources(ID3D11DeviceContext* d3dDeviceContext,
                              ID3D11ShaderReflection* shaderReflection,
                              const char *name,
                              UINT count,
                              ID3D11ShaderResourceView** SRVs);
    void PSSetShaderResources(ID3D11DeviceContext* d3dDeviceContext,
                              ID3D11ShaderReflection* shaderReflection,
                              const char *name,
                              UINT count,
                              ID3D11ShaderResourceView** SRVs);
    void CSSetShaderResources(ID3D11DeviceContext* d3dDeviceContext,
                              ID3D11ShaderReflection* shaderReflection,
                              const char *name,
                              UINT count,
                              ID3D11ShaderResourceView** SRVs);

    void VSSetSamplers(ID3D11DeviceContext* d3dDeviceContext,
                       ID3D11ShaderReflection* shaderReflection,
                       const char *name,
                       UINT count,
                       ID3D11SamplerState** samplers);
    void GSSetSamplers(ID3D11DeviceContext* d3dDeviceContext,
                       ID3D11ShaderReflection* shaderReflection,
                       const char *name,
                       UINT count,
                       ID3D11SamplerState** samplers);
    void PSSetSamplers(ID3D11DeviceContext* d3dDeviceContext,
                       ID3D11ShaderReflection* shaderReflection,
                       const char *name,
                       UINT count,
                       ID3D11SamplerState** samplers);
    void CSSetSamplers(ID3D11DeviceContext* d3dDeviceContext,
                       ID3D11ShaderReflection* shaderReflection,
                       const char *name,
                       UINT count,
                       ID3D11SamplerState** samplers);

    void CSSetUnorderedAccessViews(ID3D11DeviceContext* d3dDeviceContext,
                                   ID3D11ShaderReflection* shaderReflection,
                                   const char *name,
                                   UINT count,
                                   ID3D11UnorderedAccessView **UAVs,
                                   const UINT *UAVInitialCounts);

    HRESULT GetBindIndex(ID3D11ShaderReflection* shaderReflection, 
                         const char *name,
                         UINT *bindIndex);
    UINT GetStartBindIndex(ID3D11ShaderReflection* shaderReflection,
                           const char *paramUAVs[],
                           const UINT numUAVs);
}

#endif // H_SHADER_UTILS
