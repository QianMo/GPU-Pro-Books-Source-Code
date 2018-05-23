//--------------------------------------------------------------------------------------
// Copyright 2013 Intel Corporation
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
//--------------------------------------------------------------------------------------
#ifndef __CPUTASSETLIBRARYDX11_H__
#define __CPUTASSETLIBRARYDX11_H__

#include "CPUTAssetLibrary.h"
#include "CPUTConfigBlock.h"

#include <d3d11.h>
#include <D3DX11tex.h> // for D3DX11_IMAGE_LOAD_INFO structs

class CPUTAssetSet;
class CPUTMaterial;
class CPUTModel;
class CPUTNullNode;
class CPUTCamera;
class CPUTRenderStateBlock;
class CPUTLight;
class CPUTTexture;
class CPUTVertexShaderDX11;
class CPUTPixelShaderDX11;
class CPUTComputeShaderDX11;
class CPUTGeometryShaderDX11;
class CPUTHullShaderDX11;
class CPUTDomainShaderDX11;

//-----------------------------------------------------------------------------
struct CPUTSRGBLoadFlags
{
    bool bInterpretInputasSRGB;
    bool bWritetoSRGBOutput;
};

//-----------------------------------------------------------------------------
class CPUTAssetLibraryDX11:public CPUTAssetLibrary
{
protected:
    static CPUTAssetListEntry  *mpPixelShaderList;
    static CPUTAssetListEntry  *mpComputeShaderList;
    static CPUTAssetListEntry  *mpVertexShaderList;
    static CPUTAssetListEntry  *mpGeometryShaderList;
    static CPUTAssetListEntry  *mpHullShaderList;
    static CPUTAssetListEntry  *mpDomainShaderList;

public:
    CPUTAssetLibraryDX11(){}
    virtual ~CPUTAssetLibraryDX11()
    {
        ReleaseAllLibraryLists();
    }

    virtual void ReleaseAllLibraryLists();
    void ReleaseIunknownList( CPUTAssetListEntry *pList );

    void AddPixelShader(    const cString &name, CPUTPixelShaderDX11    *pShader) { AddAsset( name, pShader, &mpPixelShaderList ); }
    void AddComputeShader(  const cString &name, CPUTComputeShaderDX11  *pShader) { AddAsset( name, pShader, &mpComputeShaderList ); }
    void AddVertexShader(   const cString &name, CPUTVertexShaderDX11   *pShader) { AddAsset( name, pShader, &mpVertexShaderList ); }
    void AddGeometryShader( const cString &name, CPUTGeometryShaderDX11 *pShader) { AddAsset( name, pShader, &mpGeometryShaderList ); }
    void AddHullShader(     const cString &name, CPUTHullShaderDX11     *pShader) { AddAsset( name, pShader, &mpHullShaderList ); }
    void AddDomainShader(   const cString &name, CPUTDomainShaderDX11   *pShader) { AddAsset( name, pShader, &mpDomainShaderList ); }
    
    CPUTPixelShaderDX11    *FindPixelShader(    const cString &name, bool nameIsFullPathAndFilename=false ) { return    (CPUTPixelShaderDX11*)FindAsset( name, mpPixelShaderList,    nameIsFullPathAndFilename ); }
    CPUTComputeShaderDX11  *FindComputeShader(  const cString &name, bool nameIsFullPathAndFilename=false ) { return  (CPUTComputeShaderDX11*)FindAsset( name, mpComputeShaderList,  nameIsFullPathAndFilename ); }
    CPUTVertexShaderDX11   *FindVertexShader(   const cString &name, bool nameIsFullPathAndFilename=false ) { return   (CPUTVertexShaderDX11*)FindAsset( name, mpVertexShaderList,   nameIsFullPathAndFilename ); }
    CPUTGeometryShaderDX11 *FindGeometryShader( const cString &name, bool nameIsFullPathAndFilename=false ) { return (CPUTGeometryShaderDX11*)FindAsset( name, mpGeometryShaderList, nameIsFullPathAndFilename ); }
    CPUTHullShaderDX11     *FindHullShader(     const cString &name, bool nameIsFullPathAndFilename=false ) { return     (CPUTHullShaderDX11*)FindAsset( name, mpHullShaderList,     nameIsFullPathAndFilename ); }
    CPUTDomainShaderDX11   *FindDomainShader(   const cString &name, bool nameIsFullPathAndFilename=false ) { return   (CPUTDomainShaderDX11*)FindAsset( name, mpDomainShaderList,   nameIsFullPathAndFilename ); }

    // shaders - vertex, pixel
    CPUTResult GetPixelShader(     const cString &name, ID3D11Device *pD3dDevice, const cString &shaderMain, const cString &shaderProfile, CPUTPixelShaderDX11    **ppShader, bool nameIsFullPathAndFilename=false);
    CPUTResult GetComputeShader(   const cString &name, ID3D11Device *pD3dDevice, const cString &shaderMain, const cString &shaderProfile, CPUTComputeShaderDX11  **ppShader, bool nameIsFullPathAndFilename=false);
    CPUTResult GetVertexShader(    const cString &name, ID3D11Device *pD3dDevice, const cString &shaderMain, const cString &shaderProfile, CPUTVertexShaderDX11   **ppShader, bool nameIsFullPathAndFilename=false);
    CPUTResult GetGeometryShader(  const cString &name, ID3D11Device *pD3dDevice, const cString &shaderMain, const cString &shaderProfile, CPUTGeometryShaderDX11 **ppShader, bool nameIsFullPathAndFilename=false);
    CPUTResult GetHullShader(      const cString &name, ID3D11Device *pD3dDevice, const cString &shaderMain, const cString &shaderProfile, CPUTHullShaderDX11     **ppShader, bool nameIsFullPathAndFilename=false);
    CPUTResult GetDomainShader(    const cString &name, ID3D11Device *pD3dDevice, const cString &shaderMain, const cString &shaderProfile, CPUTDomainShaderDX11   **ppShader, bool nameIsFullPathAndFilename=false);
 
    // shaders - vertex, pixel
    CPUTResult CreatePixelShaderFromMemory(     const cString &name, ID3D11Device *pD3dDevice, const cString &shaderMain, const cString &shaderProfile, CPUTPixelShaderDX11    **ppShader, char *pShaderSource );
    CPUTResult CreateComputeShaderFromMemory(   const cString &name, ID3D11Device *pD3dDevice, const cString &shaderMain, const cString &shaderProfile, CPUTComputeShaderDX11  **ppShader, char *pShaderSource );
    CPUTResult CreateVertexShaderFromMemory(    const cString &name, ID3D11Device *pD3dDevice, const cString &shaderMain, const cString &shaderProfile, CPUTVertexShaderDX11   **ppShader, char *pShaderSource );
    CPUTResult CreateGeometryShaderFromMemory(  const cString &name, ID3D11Device *pD3dDevice, const cString &shaderMain, const cString &shaderProfile, CPUTGeometryShaderDX11 **ppShader, char *pShaderSource );
    CPUTResult CreateHullShaderFromMemory(      const cString &name, ID3D11Device *pD3dDevice, const cString &shaderMain, const cString &shaderProfile, CPUTHullShaderDX11     **ppShader, char *pShaderSource );
    CPUTResult CreateDomainShaderFromMemory(    const cString &name, ID3D11Device *pD3dDevice, const cString &shaderMain, const cString &shaderProfile, CPUTDomainShaderDX11   **ppShader, char *pShaderSource );
 
    CPUTResult CompileShaderFromFile(  const cString &fileName,   const cString &shaderMain, const cString &shaderProfile, ID3DBlob **ppBlob);
    CPUTResult CompileShaderFromMemory(const char *pShaderSource, const cString &shaderMain, const cString &shaderProfile, ID3DBlob **ppBlob);
};

#endif // #ifndef __CPUTASSETLIBRARYDX11_H__
