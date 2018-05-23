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
#ifndef __CPUTMATERIALDX11_H__
#define __CPUTMATERIALDX11_H__

#include <d3d11.h>

#include "CPUTMaterial.h"
class CPUTPixelShaderDX11;
class CPUTComputeShaderDX11;
class CPUTVertexShaderDX11;
class CPUTGeometryShaderDX11;
class CPUTHullShaderDX11;
class CPUTDomainShaderDX11;

class CPUTShaderParameters
{
public:
    UINT                       mTextureCount;
    cString                   *mpTextureParameterName;
    UINT                      *mpTextureParameterBindPoint;
    UINT                       mTextureParameterCount;
    CPUTTexture               *mpTexture[CPUT_MATERIAL_MAX_TEXTURE_SLOTS];
    CPUTBuffer                *mpBuffer[CPUT_MATERIAL_MAX_BUFFER_SLOTS];
    CPUTBuffer                *mpUAV[CPUT_MATERIAL_MAX_UAV_SLOTS];
    CPUTBuffer                *mpConstantBuffer[CPUT_MATERIAL_MAX_CONSTANT_BUFFER_SLOTS];

    cString                   *mpSamplerParameterName;
    UINT                      *mpSamplerParameterBindPoint;
    UINT                       mSamplerParameterCount;

    UINT                       mBufferCount;
    UINT                       mBufferParameterCount;
    cString                   *mpBufferParameterName;
    UINT                      *mpBufferParameterBindPoint;

    UINT                       mUAVCount;
    UINT                       mUAVParameterCount;
    cString                   *mpUAVParameterName;
    UINT                      *mpUAVParameterBindPoint;

    UINT                       mConstantBufferCount;
    UINT                       mConstantBufferParameterCount;
    cString                   *mpConstantBufferParameterName;
    UINT                      *mpConstantBufferParameterBindPoint;

    UINT                       mBindViewMin;
    UINT                       mBindViewMax;

    UINT                       mBindUAVMin;
    UINT                       mBindUAVMax;

    UINT                       mBindConstantBufferMin;
    UINT                       mBindConstantBufferMax;

    ID3D11ShaderResourceView  *mppBindViews[CPUT_MATERIAL_MAX_SRV_SLOTS];
    ID3D11UnorderedAccessView *mppBindUAVs[CPUT_MATERIAL_MAX_UAV_SLOTS];
    ID3D11Buffer              *mppBindConstantBuffers[CPUT_MATERIAL_MAX_CONSTANT_BUFFER_SLOTS];

    CPUTShaderParameters() :
        mTextureCount(0),
        mTextureParameterCount(0),
        mpTextureParameterName(NULL),
        mpTextureParameterBindPoint(NULL),
        mSamplerParameterCount(0),
        mpSamplerParameterName(NULL),
        mpSamplerParameterBindPoint(NULL),
        mBufferCount(0),
        mBufferParameterCount(0),
        mpBufferParameterName(NULL),
        mpBufferParameterBindPoint(NULL),
        mUAVCount(0),
        mUAVParameterCount(0),
        mpUAVParameterName(NULL),
        mpUAVParameterBindPoint(NULL),
        mConstantBufferCount(0),
        mConstantBufferParameterCount(0),
        mpConstantBufferParameterName(NULL),
        mpConstantBufferParameterBindPoint(NULL),
        mBindViewMin(CPUT_MATERIAL_MAX_SRV_SLOTS),
        mBindViewMax(0),
        mBindUAVMin(CPUT_MATERIAL_MAX_UAV_SLOTS),
        mBindUAVMax(0),
        mBindConstantBufferMin(CPUT_MATERIAL_MAX_CONSTANT_BUFFER_SLOTS),
        mBindConstantBufferMax(0)
    {
        // initialize texture slot list to null
        for(int ii=0; ii<CPUT_MATERIAL_MAX_TEXTURE_SLOTS; ii++)
        {
            mppBindViews[ii] = NULL;
            mpTexture[ii] = NULL;
        }
        for(int ii=0; ii<CPUT_MATERIAL_MAX_BUFFER_SLOTS; ii++)
        {
            mpBuffer[ii] = NULL;
        }
        for(int ii=0; ii<CPUT_MATERIAL_MAX_UAV_SLOTS; ii++)
        {
            mppBindUAVs[ii] = NULL;
            mpUAV[ii] = NULL;
        }
        for(int ii=0; ii<CPUT_MATERIAL_MAX_CONSTANT_BUFFER_SLOTS; ii++)
        {
            mppBindConstantBuffers[ii] = NULL;
            mpConstantBuffer[ii] = NULL;
        }
    };
    ~CPUTShaderParameters();
    void CloneShaderParameters( CPUTShaderParameters *pShaderParameter );
};

static const int CPUT_NUM_SHADER_PARAMETER_LISTS = 7;
class CPUTMaterialDX11 : public CPUTMaterial
{
protected:
    static void *mpLastVertexShader;
    static void *mpLastPixelShader;
    static void *mpLastComputeShader;
    static void *mpLastGeometryShader;
    static void *mpLastHullShader;
    static void *mpLastDomainShader;

    static void *mpLastVertexShaderViews[CPUT_MATERIAL_MAX_TEXTURE_SLOTS];
    static void *mpLastPixelShaderViews[CPUT_MATERIAL_MAX_TEXTURE_SLOTS];
    static void *mpLastComputeShaderViews[CPUT_MATERIAL_MAX_TEXTURE_SLOTS];
    static void *mpLastGeometryShaderViews[CPUT_MATERIAL_MAX_TEXTURE_SLOTS];
    static void *mpLastHullShaderViews[CPUT_MATERIAL_MAX_TEXTURE_SLOTS];
    static void *mpLastDomainShaderViews[CPUT_MATERIAL_MAX_TEXTURE_SLOTS];

    static void *mpLastVertexShaderConstantBuffers[CPUT_MATERIAL_MAX_CONSTANT_BUFFER_SLOTS];
    static void *mpLastPixelShaderConstantBuffers[CPUT_MATERIAL_MAX_CONSTANT_BUFFER_SLOTS];
    static void *mpLastComputeShaderConstantBuffers[CPUT_MATERIAL_MAX_CONSTANT_BUFFER_SLOTS];
    static void *mpLastGeometryShaderConstantBuffers[CPUT_MATERIAL_MAX_CONSTANT_BUFFER_SLOTS];
    static void *mpLastHullShaderConstantBuffers[CPUT_MATERIAL_MAX_CONSTANT_BUFFER_SLOTS];
    static void *mpLastDomainShaderConstantBuffers[CPUT_MATERIAL_MAX_CONSTANT_BUFFER_SLOTS];

    static void *mpLastComputeShaderUAVs[CPUT_MATERIAL_MAX_UAV_SLOTS];

    static void *mpLastRenderStateBlock;

    // Each material references materials.  A multiple-submaterial material leaves these as NULL.
    CPUTPixelShaderDX11    *mpPixelShader;
    CPUTComputeShaderDX11  *mpComputeShader; // TODO: Do Compute Shaders belong in material?
    CPUTVertexShaderDX11   *mpVertexShader;
    CPUTGeometryShaderDX11 *mpGeometryShader;
    CPUTHullShaderDX11     *mpHullShader;
    CPUTDomainShaderDX11   *mpDomainShader;

public:
    CPUTShaderParameters    mPixelShaderParameters;
    CPUTShaderParameters    mComputeShaderParameters;
    CPUTShaderParameters    mVertexShaderParameters;
    CPUTShaderParameters    mGeometryShaderParameters;
    CPUTShaderParameters    mHullShaderParameters;
    CPUTShaderParameters    mDomainShaderParameters;
    CPUTShaderParameters   *mpShaderParametersList[CPUT_NUM_SHADER_PARAMETER_LISTS]; // Constructor initializes this as a list of pointers to the above shader parameters.

protected:

    ~CPUTMaterialDX11();  // Destructor is not public.  Must release instead of delete.

    void ReadShaderSamplersAndTextures(   ID3DBlob *pBlob, CPUTShaderParameters *pShaderParameter );

    void BindTextures(        CPUTShaderParameters &params, const cString &modelSuffix, const cString &meshSuffix );
    void BindBuffers(         CPUTShaderParameters &params, const cString &modelSuffix, const cString &meshSuffix );
    void BindUAVs(            CPUTShaderParameters &params, const cString &modelSuffix, const cString &meshSuffix );
    void BindConstantBuffers( CPUTShaderParameters &params, const cString &modelSuffix, const cString &meshSuffix );

public:
    CPUTMaterialDX11();

    CPUTResult    LoadMaterial(const cString &fileName, const cString &modelSuffix, const cString &meshSuffix);
    void          ReleaseTexturesAndBuffers();
    void          RebindTexturesAndBuffers();
    CPUTVertexShaderDX11   *GetVertexShader()   { return mpVertexShader; }
    CPUTPixelShaderDX11    *GetPixelShader()    { return mpPixelShader; }
    CPUTGeometryShaderDX11 *GetGeometryShader() { return mpGeometryShader; }
    CPUTComputeShaderDX11  *GetComputeShader()  { return mpComputeShader; }
    CPUTDomainShaderDX11   *GetDomainShader()   { return mpDomainShader; }
    CPUTHullShaderDX11     *GetHullShader()     { return mpHullShader; }

    // Tells material to set the current render state to match the properties, textures,
    //  shaders, state, etc that this material represents
    void SetRenderStates( CPUTRenderParameters &renderParams );
    bool MaterialRequiresPerModelPayload();
    CPUTMaterial *CloneMaterial( const cString &absolutePathAndFilename, const cString &modelSuffix, const cString &meshSuffix );
    static void ResetStateTracking()
    {
        mpLastVertexShader = (void*)-1;
        mpLastPixelShader = (void*)-1;
        mpLastComputeShader = (void*)-1;
        mpLastGeometryShader = (void*)-1;
        mpLastHullShader = (void*)-1;
        mpLastDomainShader = (void*)-1;
        mpLastRenderStateBlock = (void*)-1;
        for(UINT ii=0; ii<CPUT_MATERIAL_MAX_TEXTURE_SLOTS; ii++ )
        {
            mpLastVertexShaderViews[ii] = (void*)-1;
            mpLastPixelShaderViews[ii] = (void*)-1;
            mpLastComputeShaderViews[ii] = (void*)-1;
            mpLastGeometryShaderViews[ii] = (void*)-1;
            mpLastHullShaderViews[ii] = (void*)-1;
            mpLastDomainShaderViews[ii] = (void*)-1;
        }
        for(UINT ii=0; ii<CPUT_MATERIAL_MAX_CONSTANT_BUFFER_SLOTS; ii++ )
        {
            mpLastVertexShaderConstantBuffers[ii]   = (void*)-1;
            mpLastPixelShaderConstantBuffers[ii]    = (void*)-1;
            mpLastComputeShaderConstantBuffers[ii]  = (void*)-1;
            mpLastGeometryShaderConstantBuffers[ii] = (void*)-1;
            mpLastHullShaderConstantBuffers[ii]     = (void*)-1;
            mpLastDomainShaderConstantBuffers[ii]   = (void*)-1;
        }
        for(UINT ii=0; ii<CPUT_MATERIAL_MAX_UAV_SLOTS; ii++ )
        {
            mpLastComputeShaderUAVs[ii] = (void*)-1;
        }
    }

};

#endif // #ifndef __CPUTMATERIALDX11_H__
