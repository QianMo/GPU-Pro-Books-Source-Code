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
#include "CPUTMaterialDX11.h"
#include "CPUT_DX11.h"
#include "CPUTTextureDX11.h"
#include "CPUTBufferDX11.h"
#include "CPUTVertexShaderDX11.h"
#include "CPUTPixelShaderDX11.h"
#include "CPUTComputeShaderDX11.h"
#include "CPUTGeometryShaderDX11.h"
#include "CPUTDomainShaderDX11.h"
#include "CPUTHullShaderDX11.h"

#define OUTPUT_BINDING_DEBUG_INFO(x)
// #define OUTPUT_BINDING_DEBUG_INFO(x) OutputDebugString(x)

CPUTConfigBlock CPUTMaterial::mGlobalProperties;

// Note: These initial values shouldn't really matter.  We call ResetStateTracking() before we render (and it performs these initializations)
void *CPUTMaterialDX11::mpLastVertexShader   = (void*)-1;
void *CPUTMaterialDX11::mpLastPixelShader    = (void*)-1;
void *CPUTMaterialDX11::mpLastComputeShader  = (void*)-1;
void *CPUTMaterialDX11::mpLastGeometryShader = (void*)-1;
void *CPUTMaterialDX11::mpLastHullShader     = (void*)-1;
void *CPUTMaterialDX11::mpLastDomainShader   = (void*)-1;
void *CPUTMaterialDX11::mpLastVertexShaderViews[CPUT_MATERIAL_MAX_TEXTURE_SLOTS]                     = {0};
void *CPUTMaterialDX11::mpLastPixelShaderViews[CPUT_MATERIAL_MAX_TEXTURE_SLOTS]                      = {0};
void *CPUTMaterialDX11::mpLastComputeShaderViews[CPUT_MATERIAL_MAX_TEXTURE_SLOTS]                    = {0};
void *CPUTMaterialDX11::mpLastGeometryShaderViews[CPUT_MATERIAL_MAX_TEXTURE_SLOTS]                   = {0};
void *CPUTMaterialDX11::mpLastHullShaderViews[CPUT_MATERIAL_MAX_TEXTURE_SLOTS]                       = {0};
void *CPUTMaterialDX11::mpLastDomainShaderViews[CPUT_MATERIAL_MAX_TEXTURE_SLOTS]                     = {0};
void *CPUTMaterialDX11::mpLastVertexShaderConstantBuffers[CPUT_MATERIAL_MAX_CONSTANT_BUFFER_SLOTS]   = {0};
void *CPUTMaterialDX11::mpLastPixelShaderConstantBuffers[CPUT_MATERIAL_MAX_CONSTANT_BUFFER_SLOTS]    = {0};
void *CPUTMaterialDX11::mpLastComputeShaderConstantBuffers[CPUT_MATERIAL_MAX_CONSTANT_BUFFER_SLOTS]  = {0};
void *CPUTMaterialDX11::mpLastGeometryShaderConstantBuffers[CPUT_MATERIAL_MAX_CONSTANT_BUFFER_SLOTS] = {0};
void *CPUTMaterialDX11::mpLastHullShaderConstantBuffers[CPUT_MATERIAL_MAX_CONSTANT_BUFFER_SLOTS]     = {0};
void *CPUTMaterialDX11::mpLastDomainShaderConstantBuffers[CPUT_MATERIAL_MAX_CONSTANT_BUFFER_SLOTS]   = {0};
void *CPUTMaterialDX11::mpLastComputeShaderUAVs[CPUT_MATERIAL_MAX_UAV_SLOTS]                         = {0};
void *CPUTMaterialDX11::mpLastRenderStateBlock  = (void*)-1;

//-----------------------------------------------------------------------------
CPUTShaderParameters::~CPUTShaderParameters()
{
    for(int ii=0; ii<CPUT_MATERIAL_MAX_TEXTURE_SLOTS; ii++)
    {
        SAFE_RELEASE(mppBindViews[ii]);
        SAFE_RELEASE(mpTexture[ii]);
    }
    for(int ii=0; ii<CPUT_MATERIAL_MAX_BUFFER_SLOTS; ii++)
    {
        SAFE_RELEASE(mpBuffer[ii]);
    }
    for(int ii=0; ii<CPUT_MATERIAL_MAX_UAV_SLOTS; ii++)
    {
        SAFE_RELEASE(mppBindUAVs[ii]);
        SAFE_RELEASE(mpUAV[ii]);
    }
    for(int ii=0; ii<CPUT_MATERIAL_MAX_CONSTANT_BUFFER_SLOTS; ii++)
    {
        SAFE_RELEASE(mppBindConstantBuffers[ii]);
        SAFE_RELEASE(mpConstantBuffer[ii]);
    }
    SAFE_DELETE_ARRAY(mpTextureParameterName);
    SAFE_DELETE_ARRAY(mpTextureParameterBindPoint);
    SAFE_DELETE_ARRAY(mpSamplerParameterName);
    SAFE_DELETE_ARRAY(mpSamplerParameterBindPoint);
    SAFE_DELETE_ARRAY(mpBufferParameterName);
    SAFE_DELETE_ARRAY(mpBufferParameterBindPoint);
    SAFE_DELETE_ARRAY(mpUAVParameterName);
    SAFE_DELETE_ARRAY(mpUAVParameterBindPoint);
    SAFE_DELETE_ARRAY(mpConstantBufferParameterName);
    SAFE_DELETE_ARRAY(mpConstantBufferParameterBindPoint)
}

//-----------------------------------------------------------------------------
CPUTMaterialDX11::CPUTMaterialDX11() :
    mpPixelShader(NULL),
    mpComputeShader(NULL),
    mpVertexShader(NULL),
    mpGeometryShader(NULL),
    mpHullShader(NULL),
    mpDomainShader(NULL)
{
    // TODO: Is there a better/safer way to initialize this list?
    mpShaderParametersList[0] =  &mPixelShaderParameters,
    mpShaderParametersList[1] =  &mComputeShaderParameters,
    mpShaderParametersList[2] =  &mVertexShaderParameters,
    mpShaderParametersList[3] =  &mGeometryShaderParameters,
    mpShaderParametersList[4] =  &mHullShaderParameters,
    mpShaderParametersList[5] =  &mDomainShaderParameters,
    mpShaderParametersList[6] =  NULL;
}

//-----------------------------------------------------------------------------
CPUTMaterialDX11::~CPUTMaterialDX11()
{
    SAFE_RELEASE(mpPixelShader);
    SAFE_RELEASE(mpComputeShader);
    SAFE_RELEASE(mpVertexShader);
    SAFE_RELEASE(mpGeometryShader);
    SAFE_RELEASE(mpHullShader);
    SAFE_RELEASE(mpDomainShader);
    SAFE_RELEASE(mpRenderStateBlock);

    CPUTMaterial::~CPUTMaterial();
}

// **********************************
// **** Set Shader resources if they changed
// **********************************
#define SET_SHADER_RESOURCES( SHADER, SHADER_TYPE ) \
    /* If the shader changed ... */ \
    if( mpLast##SHADER##Shader != mp##SHADER##Shader ) \
    { \
        mpLast##SHADER##Shader = mp##SHADER##Shader; \
        pContext->##SHADER_TYPE##SetShader( mp##SHADER##Shader ? mp##SHADER##Shader->GetNative##SHADER##Shader() : NULL, NULL, 0 ); \
    } \
    /* Spend time checking shader resources only if a shader is bound ... */ \
    if( mp##SHADER##Shader ) \
    { \
        if( m##SHADER##ShaderParameters.mTextureCount || m##SHADER##ShaderParameters.mBufferCount ) \
        { \
            same = true; \
            /* If all of the texture slots we need are already bound to our textures, then skip setting the SRVs... */\
            for( UINT ii=0; ii < m##SHADER##ShaderParameters.mTextureCount; ii++ ) \
            { \
                UINT bindPoint = m##SHADER##ShaderParameters.mpTextureParameterBindPoint[ii]; \
                if(mpLast##SHADER##ShaderViews[bindPoint] != m##SHADER##ShaderParameters.mppBindViews[bindPoint] ) \
                { \
                    mpLast##SHADER##ShaderViews[bindPoint] = m##SHADER##ShaderParameters.mppBindViews[bindPoint]; \
                    same = false; \
                } \
            } \
            for( UINT ii=0; ii < m##SHADER##ShaderParameters.mBufferCount; ii++ ) \
            { \
                UINT bindPoint = m##SHADER##ShaderParameters.mpBufferParameterBindPoint[ii]; \
                if(mpLast##SHADER##ShaderViews[bindPoint] != m##SHADER##ShaderParameters.mppBindViews[bindPoint] ) \
                { \
                    mpLast##SHADER##ShaderViews[bindPoint] = m##SHADER##ShaderParameters.mppBindViews[bindPoint]; \
                    same = false; \
                } \
            } \
            if( !same ) \
            { \
                int min   = m##SHADER##ShaderParameters.mBindViewMin; \
                int max   = m##SHADER##ShaderParameters.mBindViewMax; \
                int count = max - min + 1; \
                pContext->SHADER_TYPE##SetShaderResources( min, count, &m##SHADER##ShaderParameters.mppBindViews[min] ); \
            } \
        } \
        if( m##SHADER##ShaderParameters.mConstantBufferCount ) \
        { \
            same = true; \
            /* If all of the constant buffer slots we need are already bound to our constant buffers, then skip setting the SRVs... */\
            for( UINT ii=0; ii<m##SHADER##ShaderParameters.mConstantBufferCount; ii++ ) \
            { \
                UINT bindPoint = m##SHADER##ShaderParameters.mpConstantBufferParameterBindPoint[ii]; \
                if(mpLast##SHADER##ShaderConstantBuffers[bindPoint] != m##SHADER##ShaderParameters.mppBindConstantBuffers[bindPoint] ) \
                { \
                    mpLast##SHADER##ShaderConstantBuffers[bindPoint] = m##SHADER##ShaderParameters.mppBindConstantBuffers[bindPoint]; \
                    same = false; \
                } \
            } \
            if(!same) \
            { \
                int min   = m##SHADER##ShaderParameters.mBindConstantBufferMin; \
                int max   = m##SHADER##ShaderParameters.mBindConstantBufferMax; \
                int count = max - min + 1; \
                pContext->SHADER_TYPE##SetConstantBuffers(min, count,    &m##SHADER##ShaderParameters.mppBindConstantBuffers[min] ); \
            } \
        } \
    }

//-----------------------------------------------------------------------------
void CPUTMaterialDX11::SetRenderStates( CPUTRenderParameters &renderParams )
{
    ID3D11DeviceContext *pContext = ((CPUTRenderParametersDX*)&renderParams)->mpContext;

    bool same = true;

    SET_SHADER_RESOURCES( Vertex,    VS );
    SET_SHADER_RESOURCES( Pixel,     PS );
    SET_SHADER_RESOURCES( Compute,   CS );
    SET_SHADER_RESOURCES( Geometry,  GS );
    SET_SHADER_RESOURCES( Hull,      HS );
    SET_SHADER_RESOURCES( Domain,    DS );

    // Only the compute shader may have UAVs to bind.
    // Note that pixel shaders can too, but DX requires setting those when setting RTV(s).
    same = true;
    for( UINT ii=0; ii<mComputeShaderParameters.mUAVCount; ii++ )
    {
        UINT bindPoint = mComputeShaderParameters.mpUAVParameterBindPoint[ii];
        if(mpLastComputeShaderUAVs[ii] != mComputeShaderParameters.mppBindUAVs[bindPoint] )
        {
            mpLastComputeShaderUAVs[ii] = mComputeShaderParameters.mppBindUAVs[bindPoint];
            same = false;
        }
    }
#if 0
	if( mPixelShaderParameters.mUAVCount )
	{
        ID3D11RenderTargetView *pRTVs[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT];
        ID3D11DepthStencilView *pDSV;
        pContext->OMGetRenderTargets( D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, &pRTVs[0], &pDSV );
        // Count the RTVs
        UINT rtvCount;
        for( rtvCount=D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT-1; rtvCount>0; rtvCount-- )
        {
            if( 0 != pRTVs[rtvCount] ) break;
        }
        rtvCount+=1;
	    pContext->OMSetRenderTargetsAndUnorderedAccessViews(
            rtvCount,
            pRTVs,
            pDSV,
            rtvCount,
            CPUT_MATERIAL_MAX_UAV_SLOTS-rtvCount,
            &mPixelShaderParameters.mppBindUAVs[rtvCount],
            NULL
        );
        SAFE_RELEASE(pDSV);
        for( UINT ii=0; ii<D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; ii++ )
        {
            SAFE_RELEASE( pRTVs[ii] );
        }
	}
#endif
    if(mComputeShaderParameters.mUAVCount && !same)
    {
        int min   = mComputeShaderParameters.mBindUAVMin;
        int max   = mComputeShaderParameters.mBindUAVMax;
        int count = max - min + 1;
        pContext->CSSetUnorderedAccessViews(min, count, &mComputeShaderParameters.mppBindUAVs[min], NULL );
    }

    // Set the render state block if it changed
    if( mpLastRenderStateBlock != mpRenderStateBlock )
    {
        mpLastRenderStateBlock = mpRenderStateBlock;
        if( mpRenderStateBlock )
        {
            // We know we have a DX11 class.  Does this correctly bypass the virtual?
            // Should we move it to the DX11 class.
            ((CPUTRenderStateBlockDX11*)mpRenderStateBlock)->SetRenderStates(renderParams);
        }
        else
        {
            CPUTRenderStateBlock::GetDefaultRenderStateBlock()->SetRenderStates(renderParams);
        }
    }
}

//-----------------------------------------------------------------------------
void CPUTMaterialDX11::ReadShaderSamplersAndTextures( ID3DBlob *pBlob, CPUTShaderParameters *pShaderParameter )
{
    // ***************************
    // Use shader reflection to get texture and sampler names.  We use them later to bind .mtl texture-specification to shader parameters/variables.
    // TODO: Currently do this only for PS.  Do for other shader types too.
    // TODO: Generalize, so easy to call for different shader types
    // ***************************
    ID3D11ShaderReflection *pReflector = NULL; 
    D3D11_SHADER_INPUT_BIND_DESC desc;

    D3DReflect( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), IID_ID3D11ShaderReflection, (void**)&pReflector);
    // Walk through the shader input bind descriptors.  Find the samplers and textures.
    int ii=0;
	HRESULT hr = pReflector->GetResourceBindingDesc( ii++, &desc );
	while( SUCCEEDED(hr) )
	{
        switch( desc.Type )
        {
        case D3D_SIT_TEXTURE:
            pShaderParameter->mTextureParameterCount++;
            break;
        case D3D_SIT_SAMPLER:
            pShaderParameter->mSamplerParameterCount++;
            break;
        case D3D_SIT_CBUFFER:
            pShaderParameter->mConstantBufferParameterCount++;
            break;

        case D3D_SIT_TBUFFER:
        case D3D_SIT_STRUCTURED:
        case D3D_SIT_BYTEADDRESS:
            pShaderParameter->mBufferParameterCount++;
            break;

        case D3D_SIT_UAV_RWTYPED:
        case D3D_SIT_UAV_RWSTRUCTURED:
        case D3D_SIT_UAV_RWBYTEADDRESS:
        case D3D_SIT_UAV_APPEND_STRUCTURED:
        case D3D_SIT_UAV_CONSUME_STRUCTURED:
        case D3D_SIT_UAV_RWSTRUCTURED_WITH_COUNTER:
            pShaderParameter->mUAVParameterCount++;
            break;
        }
        hr = pReflector->GetResourceBindingDesc( ii++, &desc );
    }

    pShaderParameter->mpTextureParameterName              = new cString[pShaderParameter->mTextureParameterCount];
    pShaderParameter->mpTextureParameterBindPoint         = new UINT[   pShaderParameter->mTextureParameterCount];
    pShaderParameter->mpSamplerParameterName              = new cString[pShaderParameter->mSamplerParameterCount];
    pShaderParameter->mpSamplerParameterBindPoint         = new UINT[   pShaderParameter->mSamplerParameterCount];
    pShaderParameter->mpBufferParameterName               = new cString[pShaderParameter->mBufferParameterCount];
    pShaderParameter->mpBufferParameterBindPoint          = new UINT[   pShaderParameter->mBufferParameterCount];
    pShaderParameter->mpUAVParameterName                  = new cString[pShaderParameter->mUAVParameterCount];
    pShaderParameter->mpUAVParameterBindPoint             = new UINT[   pShaderParameter->mUAVParameterCount];
    pShaderParameter->mpConstantBufferParameterName       = new cString[pShaderParameter->mConstantBufferParameterCount];
    pShaderParameter->mpConstantBufferParameterBindPoint  = new UINT[   pShaderParameter->mConstantBufferParameterCount];
        
    // Start over.  This time, copy the names.
    ii=0;
    UINT textureIndex = 0;
    UINT samplerIndex = 0;
    UINT bufferIndex = 0;
    UINT uavIndex = 0;
    UINT constantBufferIndex = 0;
    hr = pReflector->GetResourceBindingDesc( ii++, &desc );

    while( SUCCEEDED(hr) )
    {
        switch( desc.Type )
        {
        case D3D_SIT_TEXTURE:
            pShaderParameter->mpTextureParameterName[textureIndex] = s2ws(desc.Name);
            pShaderParameter->mpTextureParameterBindPoint[textureIndex] = desc.BindPoint;
            textureIndex++;
            break;
        case D3D_SIT_SAMPLER:
            pShaderParameter->mpSamplerParameterName[samplerIndex] = s2ws(desc.Name);
            pShaderParameter->mpSamplerParameterBindPoint[samplerIndex] = desc.BindPoint;
            samplerIndex++;
            break;
        case D3D_SIT_CBUFFER:
            pShaderParameter->mpConstantBufferParameterName[constantBufferIndex] = s2ws(desc.Name);
            pShaderParameter->mpConstantBufferParameterBindPoint[constantBufferIndex] = desc.BindPoint;
            constantBufferIndex++;
            break;
        case D3D_SIT_TBUFFER:
        case D3D_SIT_STRUCTURED:
        case D3D_SIT_BYTEADDRESS:
            pShaderParameter->mpBufferParameterName[bufferIndex] = s2ws(desc.Name);
            pShaderParameter->mpBufferParameterBindPoint[bufferIndex] = desc.BindPoint;
            bufferIndex++;
            break;
        case D3D_SIT_UAV_RWTYPED:
        case D3D_SIT_UAV_RWSTRUCTURED:
        case D3D_SIT_UAV_RWBYTEADDRESS:
        case D3D_SIT_UAV_APPEND_STRUCTURED:
        case D3D_SIT_UAV_CONSUME_STRUCTURED:
        case D3D_SIT_UAV_RWSTRUCTURED_WITH_COUNTER:
            pShaderParameter->mpUAVParameterName[uavIndex] = s2ws(desc.Name);
            pShaderParameter->mpUAVParameterBindPoint[uavIndex] = desc.BindPoint;
            uavIndex++;
            break;
        }
        hr = pReflector->GetResourceBindingDesc( ii++, &desc );
    }
}

//-----------------------------------------------------------------------------
void CPUTMaterialDX11::BindTextures( CPUTShaderParameters &params, const cString &modelSuffix, const cString &meshSuffix )
{
    CPUTAssetLibraryDX11 *pAssetLibrary = (CPUTAssetLibraryDX11*)CPUTAssetLibrary::GetAssetLibrary();

    for(params.mTextureCount=0; params.mTextureCount < params.mTextureParameterCount; params.mTextureCount++)
    {
        cString textureName;
        UINT textureCount = params.mTextureCount;
        cString tagName = params.mpTextureParameterName[textureCount];
        CPUTConfigEntry *pValue = mConfigBlock.GetValueByName(tagName);
        if( !pValue->IsValid() )
        {
            // We didn't find our property in the file.  Is it in the global config block?
            pValue = mGlobalProperties.GetValueByName(tagName);
        }
        ASSERT( pValue->IsValid(), L"Can't find texture '" + tagName + L"'." ); //  TODO: fix message
        textureName = pValue->ValueAsString();
        // If the texture name not specified.  Load default.dds instead
        if( 0 == textureName.length() ) { textureName = _L("default.dds"); }

        UINT bindPoint = params.mpTextureParameterBindPoint[textureCount]; 
        ASSERT( bindPoint < CPUT_MATERIAL_MAX_TEXTURE_SLOTS, _L("Texture bind point out of range.") );

        params.mBindViewMin = min( params.mBindViewMin, bindPoint );
        params.mBindViewMax = max( params.mBindViewMax, bindPoint );

        if( textureName[0] == '@' )
        {
            // This is a per-mesh value.  Add to per-mesh list.
            textureName += modelSuffix + meshSuffix;
        } else if( textureName[0] == '#' )
        {
            // This is a per-mesh value.  Add to per-mesh list.
            textureName += modelSuffix;
        }

        // Get the sRGB flag (default to true)
        cString SRGBName = tagName+_L("sRGB");
        CPUTConfigEntry *pSRGBValue = mConfigBlock.GetValueByName(SRGBName);
        bool loadAsSRGB = pSRGBValue->IsValid() ?  loadAsSRGB = pSRGBValue->ValueAsBool() : true;

        if( !params.mpTexture[textureCount] )
        {
            params.mpTexture[textureCount] = pAssetLibrary->GetTexture( textureName, false, loadAsSRGB );
            ASSERT( params.mpTexture[textureCount], _L("Failed getting texture ") + textureName);
        }

        // The shader file (e.g. .fx) can specify the texture bind point (e.g., t0).  Those specifications 
        // might not be contiguous, and there might be gaps (bind points without assigned textures)
        // TODO: Warn about missing bind points?
        params.mppBindViews[bindPoint] = ((CPUTTextureDX11*)params.mpTexture[textureCount])->GetShaderResourceView();
        params.mppBindViews[bindPoint]->AddRef();

        OUTPUT_BINDING_DEBUG_INFO( (itoc(bindPoint) + _L(" : ") + params.mpTexture[textureCount]->GetName() + _L("\n")).c_str() );
    }
}

//-----------------------------------------------------------------------------
void CPUTMaterialDX11::BindBuffers( CPUTShaderParameters &params, const cString &modelSuffix, const cString &meshSuffix )
{
    CPUTConfigEntry *pValue;
    if( !params.mBufferParameterCount ) { return; }
    OUTPUT_BINDING_DEBUG_INFO( _L("Bound Buffers") );

    CPUTAssetLibraryDX11 *pAssetLibrary = (CPUTAssetLibraryDX11*)CPUTAssetLibrary::GetAssetLibrary();
    for(params.mBufferCount=0; params.mBufferCount < params.mBufferParameterCount; params.mBufferCount++)
    {
        cString bufferName;
        UINT bufferCount = params.mBufferCount;
        cString tagName = params.mpBufferParameterName[bufferCount];
        {
            pValue = mConfigBlock.GetValueByName(tagName);
            if( !pValue->IsValid() )
            {
                // We didn't find our property in the file.  Is it in the global config block?
                pValue = mGlobalProperties.GetValueByName(tagName);
            }
            ASSERT( pValue->IsValid(), L"Can't find buffer '" + tagName + L"'." ); //  TODO: fix message
            bufferName = pValue->ValueAsString();
        }
        UINT bindPoint = params.mpBufferParameterBindPoint[bufferCount]; 
        ASSERT( bindPoint < CPUT_MATERIAL_MAX_BUFFER_SLOTS, _L("Buffer bind point out of range.") );

        params.mBindViewMin = min( params.mBindViewMin, bindPoint );
        params.mBindViewMax = max( params.mBindViewMax, bindPoint );

        if( bufferName[0] == '@' )
        {
            // This is a per-mesh value.  Add to per-mesh list.
            bufferName += modelSuffix + meshSuffix;
        } else if( bufferName[0] == '#' )
        {
            // This is a per-mesh value.  Add to per-model list.
            bufferName += modelSuffix;
        }
        if( !params.mpBuffer[bufferCount] )
        {
            params.mpBuffer[bufferCount] = pAssetLibrary->GetBuffer( bufferName );
            ASSERT( params.mpBuffer[bufferCount], _L("Failed getting buffer ") + bufferName);
        }

        params.mppBindViews[bindPoint]   = ((CPUTBufferDX11*)params.mpBuffer[bufferCount])->GetShaderResourceView();
        if( params.mppBindViews[bindPoint] )  { params.mppBindViews[bindPoint]->AddRef();}

        OUTPUT_BINDING_DEBUG_INFO( (itoc(bindPoint) + _L(" : ") + params.mpBuffer[bufferCount]->GetName() + _L("\n")).c_str() );
    }
    OUTPUT_BINDING_DEBUG_INFO( _L("\n") );
}

//-----------------------------------------------------------------------------
void CPUTMaterialDX11::BindUAVs( CPUTShaderParameters &params, const cString &modelSuffix, const cString &meshSuffix )
{
    CPUTConfigEntry *pValue;
    if( !params.mUAVParameterCount ) { return; }
    OUTPUT_BINDING_DEBUG_INFO( _L("Bound UAVs") );

    CPUTAssetLibraryDX11 *pAssetLibrary = (CPUTAssetLibraryDX11*)CPUTAssetLibrary::GetAssetLibrary();
    memset( params.mppBindUAVs, 0, sizeof(params.mppBindUAVs) );
    for(params.mUAVCount=0; params.mUAVCount < params.mUAVParameterCount; params.mUAVCount++)
    {
        cString uavName;
        UINT uavCount = params.mUAVCount;

        cString tagName = params.mpUAVParameterName[uavCount];
        {
            pValue = mConfigBlock.GetValueByName(tagName);
            if( !pValue->IsValid() )
            {
                // We didn't find our property in the file.  Is it in the global config block?
                pValue = mGlobalProperties.GetValueByName(tagName);
            }
            ASSERT( pValue->IsValid(), L"Can't find UAV '" + tagName + L"'." ); //  TODO: fix message
            uavName = pValue->ValueAsString();
        }
        UINT bindPoint = params.mpUAVParameterBindPoint[uavCount];
        ASSERT( bindPoint < CPUT_MATERIAL_MAX_UAV_SLOTS, _L("UAV bind point out of range.") );

        params.mBindUAVMin = min( params.mBindUAVMin, bindPoint );
        params.mBindUAVMax = max( params.mBindUAVMax, bindPoint );

        if( uavName[0] == '@' )
        {
            // This is a per-mesh value.  Add to per-mesh list.
            uavName += modelSuffix + meshSuffix;
        } else if( uavName[0] == '#' )
        {
            // This is a per-mesh value.  Add to per-model list.
            uavName += modelSuffix;
        }
        if( !params.mpUAV[uavCount] )
        {
            params.mpUAV[uavCount] = pAssetLibrary->GetBuffer( uavName );
            ASSERT( params.mpUAV[uavCount], _L("Failed getting UAV ") + uavName);
        }

        // If has UAV, then add to mppBindUAV
        params.mppBindUAVs[bindPoint]   = ((CPUTBufferDX11*)params.mpUAV[uavCount])->GetUnorderedAccessView();
        if( params.mppBindUAVs[bindPoint] )  { params.mppBindUAVs[bindPoint]->AddRef();}

        OUTPUT_BINDING_DEBUG_INFO( (itoc(bindPoint) + _L(" : ") + params.mpUAV[uavCount]->GetName() + _L("\n")).c_str() );
    }
    OUTPUT_BINDING_DEBUG_INFO( _L("\n") );
}

//-----------------------------------------------------------------------------
void CPUTMaterialDX11::BindConstantBuffers( CPUTShaderParameters &params, const cString &modelSuffix, const cString &meshSuffix )
{
    CPUTConfigEntry *pValue;
    if( !params.mConstantBufferParameterCount ) { return; }
    OUTPUT_BINDING_DEBUG_INFO( _L("Bound Constant Buffers\n") );

    CPUTAssetLibraryDX11 *pAssetLibrary = (CPUTAssetLibraryDX11*)CPUTAssetLibrary::GetAssetLibrary();
    for(params.mConstantBufferCount=0; params.mConstantBufferCount < params.mConstantBufferParameterCount; params.mConstantBufferCount++)
    {
        cString constantBufferName;
        UINT constantBufferCount = params.mConstantBufferCount;

        cString tagName = params.mpConstantBufferParameterName[constantBufferCount];
        {
            pValue = mConfigBlock.GetValueByName(tagName);
            if( !pValue->IsValid() )
            {
                // We didn't find our property in the file.  Is it in the global config block?
                pValue = mGlobalProperties.GetValueByName(tagName);
            }
            ASSERT( pValue->IsValid(), L"Can't find constant buffer '" + tagName + L"'." ); //  TODO: fix message
            constantBufferName = pValue->ValueAsString();
        }
        UINT bindPoint = params.mpConstantBufferParameterBindPoint[constantBufferCount];
        ASSERT( bindPoint < CPUT_MATERIAL_MAX_CONSTANT_BUFFER_SLOTS, _L("Constant buffer bind point out of range.") );

        params.mBindConstantBufferMin = min( params.mBindConstantBufferMin, bindPoint );
        params.mBindConstantBufferMax = max( params.mBindConstantBufferMax, bindPoint );

        if( constantBufferName[0] == '@' )
        {
            constantBufferName += modelSuffix + meshSuffix;
        } else if( constantBufferName[0] == '#' )
        {
            constantBufferName += modelSuffix;
        }
        if( !params.mpConstantBuffer[constantBufferCount] )
        {
            params.mpConstantBuffer[constantBufferCount] = pAssetLibrary->GetConstantBuffer( constantBufferName );
            ASSERT( params.mpConstantBuffer[constantBufferCount], _L("Failed getting constant buffer ") + constantBufferName);
        }

        // If has constant buffer, then add to mppBindConstantBuffer
        params.mppBindConstantBuffers[bindPoint]   = ((CPUTBufferDX11*)params.mpConstantBuffer[constantBufferCount])->GetNativeBuffer();
        if( params.mppBindConstantBuffers[bindPoint] )  { params.mppBindConstantBuffers[bindPoint]->AddRef();}

        OUTPUT_BINDING_DEBUG_INFO( (itoc(bindPoint) + _L(" : ") + params.mpConstantBuffer[constantBufferCount]->GetName() + _L("\n")).c_str() );
    }
    OUTPUT_BINDING_DEBUG_INFO( _L("\n") );
}

//-----------------------------------------------------------------------------
CPUTResult CPUTMaterialDX11::LoadMaterial(const cString &fileName, const cString &modelSuffix, const cString &meshSuffix)
{
    CPUTResult result = CPUT_SUCCESS;

    mMaterialName = fileName;

    // Open/parse the file
    CPUTConfigFile file;
    result = file.LoadFile(fileName);
    if(CPUTFAILED(result))
    {
        return result;
    }

    // Make a local copy of all the parameters
	CPUTConfigBlock *pBlock = file.GetBlock(0);
    ASSERT( pBlock, _L("Error getting parameter block") );
    mConfigBlock = *pBlock;

    // get necessary device and AssetLibrary pointers
    ID3D11Device *pD3dDevice = CPUT_DX11::GetDevice();
    CPUTAssetLibraryDX11 *pAssetLibrary = (CPUTAssetLibraryDX11*)CPUTAssetLibrary::GetAssetLibrary();

    // TODO:  The following code is very repetitive.  Consider generalizing so we can call a function instead.
    // see if there are any pixel/vertex/geo shaders to load
    CPUTConfigEntry *pValue, *pEntryPointName, *pProfileName;
    pValue   = mConfigBlock.GetValueByName(_L("VertexShaderFile"));
    if( pValue->IsValid() )
    {
        pEntryPointName = mConfigBlock.GetValueByName(_L("VertexShaderMain"));
        pProfileName    = mConfigBlock.GetValueByName(_L("VertexShaderProfile"));
        pAssetLibrary->GetVertexShader(pValue->ValueAsString(), pD3dDevice, pEntryPointName->ValueAsString(), pProfileName->ValueAsString(), &mpVertexShader );
        ReadShaderSamplersAndTextures( mpVertexShader->GetBlob(), &mVertexShaderParameters );
    }

    // load and store the pixel shader if it was specified
    pValue  = mConfigBlock.GetValueByName(_L("PixelShaderFile"));
    if( pValue->IsValid() )
    {
        pEntryPointName = mConfigBlock.GetValueByName(_L("PixelShaderMain"));
        pProfileName    = mConfigBlock.GetValueByName(_L("PixelShaderProfile"));
        pAssetLibrary->GetPixelShader(pValue->ValueAsString(), pD3dDevice, pEntryPointName->ValueAsString(), pProfileName->ValueAsString(), &mpPixelShader);
        ReadShaderSamplersAndTextures( mpPixelShader->GetBlob(), &mPixelShaderParameters );
    }

    // load and store the compute shader if it was specified
    pValue = mConfigBlock.GetValueByName(_L("ComputeShaderFile"));
    if( pValue->IsValid() )
    {
        pEntryPointName = mConfigBlock.GetValueByName(_L("ComputeShaderMain"));
        pProfileName = mConfigBlock.GetValueByName(_L("ComputeShaderProfile"));
        pAssetLibrary->GetComputeShader(pValue->ValueAsString(), pD3dDevice, pEntryPointName->ValueAsString(), pProfileName->ValueAsString(), &mpComputeShader);
        ReadShaderSamplersAndTextures( mpComputeShader->GetBlob(), &mComputeShaderParameters );
    }

    // load and store the geometry shader if it was specified
    pValue = mConfigBlock.GetValueByName(_L("GeometryShaderFile"));
    if( pValue->IsValid() )
    {
        pEntryPointName = mConfigBlock.GetValueByName(_L("GeometryShaderMain"));
        pProfileName = mConfigBlock.GetValueByName(_L("GeometryShaderProfile"));
        pAssetLibrary->GetGeometryShader(pValue->ValueAsString(), pD3dDevice, pEntryPointName->ValueAsString(), pProfileName->ValueAsString(), &mpGeometryShader);
        ReadShaderSamplersAndTextures( mpGeometryShader->GetBlob(), &mGeometryShaderParameters );
    }

    // load and store the hull shader if it was specified
    pValue = mConfigBlock.GetValueByName(_L("HullShaderFile"));
    if( pValue->IsValid() )
    {
        pEntryPointName = mConfigBlock.GetValueByName(_L("HullShaderMain"));
        pProfileName = mConfigBlock.GetValueByName(_L("HullShaderProfile"));
        pAssetLibrary->GetHullShader(pValue->ValueAsString(), pD3dDevice, pEntryPointName->ValueAsString(), pProfileName->ValueAsString(), &mpHullShader);
        ReadShaderSamplersAndTextures( mpHullShader->GetBlob(), &mHullShaderParameters );
    }

    // load and store the domain shader if it was specified
    pValue = mConfigBlock.GetValueByName(_L("DomainShaderFile"));
    if( pValue->IsValid() )
    {
        pEntryPointName = mConfigBlock.GetValueByName(_L("DomainShaderMain"));
        pProfileName = mConfigBlock.GetValueByName(_L("DomainShaderProfile"));
        pAssetLibrary->GetDomainShader(pValue->ValueAsString(), pD3dDevice, pEntryPointName->ValueAsString(), pProfileName->ValueAsString(), &mpDomainShader);
        ReadShaderSamplersAndTextures( mpDomainShader->GetBlob(), &mDomainShaderParameters );
    }

    // load and store the render state file if it was specified
    pValue = mConfigBlock.GetValueByName(_L("RenderStateFile"));
    if( pValue->IsValid() )
    {
        mpRenderStateBlock = pAssetLibrary->GetRenderStateBlock(pValue->ValueAsString());
    }


    OUTPUT_BINDING_DEBUG_INFO( (_L("Bindings for : ") + mMaterialName + _L("\n")).c_str() );
    cString pShaderTypeNameList[] = {
        _L("Pixel shader"),
        _L("Compute shader"),
        _L("Vertex shader"),
        _L("Geometry shader"),
        _L("Hull shader"),
        _L("Domain shader"),
    };
    cString *pShaderTypeName = pShaderTypeNameList;

    void *pShaderList[] = {
        mpPixelShader,
        mpComputeShader,
        mpVertexShader,
        mpGeometryShader,
        mpHullShader,
        mpDomainShader
    };
    void **pShader = pShaderList;

    // For each of the shader stages, bind shaders and buffers
    for( CPUTShaderParameters **pCur = mpShaderParametersList; *pCur; pCur++ ) // Bind textures and buffersfor each shader stage
    {
        if( !*pShader++ )
        {
            pShaderTypeName++; // Increment the name pointer to remain coherent.
            continue;          // This shader not bound.  Don't waste time binding to it.
        }

        OUTPUT_BINDING_DEBUG_INFO( (*(pShaderTypeName++)  + _L("\n")).c_str() );

        BindTextures(        **pCur, modelSuffix, meshSuffix );
        BindBuffers(         **pCur, modelSuffix, meshSuffix );
        BindUAVs(            **pCur, modelSuffix, meshSuffix );
        BindConstantBuffers( **pCur, modelSuffix, meshSuffix );

        OUTPUT_BINDING_DEBUG_INFO( _L("\n") );
    }

    return result;
}

//-----------------------------------------------------------------------------
CPUTMaterial *CPUTMaterialDX11::CloneMaterial(const cString &fileName, const cString &modelSuffix, const cString &meshSuffix)
{
    CPUTMaterialDX11 *pMaterial = new CPUTMaterialDX11();

    // TODO: move texture to base class.  All APIs have textures.
    pMaterial->mpPixelShader    = mpPixelShader;    if(mpPixelShader)    mpPixelShader->AddRef();   
    pMaterial->mpComputeShader  = mpComputeShader;  if(mpComputeShader)  mpComputeShader->AddRef();
    pMaterial->mpVertexShader   = mpVertexShader;   if(mpVertexShader)   mpVertexShader->AddRef();
    pMaterial->mpGeometryShader = mpGeometryShader; if(mpGeometryShader) mpGeometryShader->AddRef();
    pMaterial->mpHullShader     = mpHullShader;     if(mpHullShader)     mpHullShader->AddRef();
    pMaterial->mpDomainShader   = mpDomainShader;   if(mpDomainShader)   mpDomainShader->AddRef();

    mPixelShaderParameters.CloneShaderParameters(    &pMaterial->mPixelShaderParameters );
    mComputeShaderParameters.CloneShaderParameters(  &pMaterial->mComputeShaderParameters );
    mVertexShaderParameters.CloneShaderParameters(   &pMaterial->mVertexShaderParameters );
    mGeometryShaderParameters.CloneShaderParameters( &pMaterial->mGeometryShaderParameters );
    mHullShaderParameters.CloneShaderParameters(     &pMaterial->mHullShaderParameters );
    mDomainShaderParameters.CloneShaderParameters(   &pMaterial->mDomainShaderParameters );

    pMaterial->mpShaderParametersList[0] =  &pMaterial->mPixelShaderParameters,
    pMaterial->mpShaderParametersList[1] =  &pMaterial->mComputeShaderParameters,
    pMaterial->mpShaderParametersList[2] =  &pMaterial->mVertexShaderParameters,
    pMaterial->mpShaderParametersList[3] =  &pMaterial->mGeometryShaderParameters,
    pMaterial->mpShaderParametersList[4] =  &pMaterial->mHullShaderParameters,
    pMaterial->mpShaderParametersList[5] =  &pMaterial->mDomainShaderParameters,
    pMaterial->mpShaderParametersList[6] =  NULL;

    pMaterial->mpRenderStateBlock = mpRenderStateBlock; if( mpRenderStateBlock ) mpRenderStateBlock->AddRef();

    pMaterial->mMaterialName      = mMaterialName;
    pMaterial->mConfigBlock       = mConfigBlock;

    // For each of the shader stages, bind shaders and buffers
    for( CPUTShaderParameters **pCur = pMaterial->mpShaderParametersList; *pCur; pCur++ ) // Bind textures and buffersfor each shader stage
    {
        pMaterial->BindTextures(        **pCur, modelSuffix, meshSuffix );
        pMaterial->BindBuffers(         **pCur, modelSuffix, meshSuffix );
        pMaterial->BindUAVs(            **pCur, modelSuffix, meshSuffix );
        pMaterial->BindConstantBuffers( **pCur, modelSuffix, meshSuffix );
    }
    return pMaterial;
}

//-----------------------------------------------------------------------------
bool CPUTMaterialDX11::MaterialRequiresPerModelPayload()
{
    return 
        (mpPixelShader    && mpPixelShader   ->ShaderRequiresPerModelPayload(mConfigBlock))  ||
        (mpComputeShader  && mpComputeShader ->ShaderRequiresPerModelPayload(mConfigBlock))  ||
        (mpVertexShader   && mpVertexShader  ->ShaderRequiresPerModelPayload(mConfigBlock))  ||
        (mpGeometryShader && mpGeometryShader->ShaderRequiresPerModelPayload(mConfigBlock))  ||
        (mpHullShader     && mpHullShader    ->ShaderRequiresPerModelPayload(mConfigBlock))  ||
        (mpDomainShader   && mpDomainShader  ->ShaderRequiresPerModelPayload(mConfigBlock));
}

//-----------------------------------------------------------------------------
void CPUTMaterialDX11::RebindTexturesAndBuffers()
{
    for( CPUTShaderParameters **pCur = mpShaderParametersList; *pCur; pCur++ ) // Rebind textures for each shader stage
    {
        for( UINT ii=0; ii<(*pCur)->mTextureCount; ii++ )
        {
            if( (*pCur)->mpTexture[ii] )
            {
                UINT bindPoint = (*pCur)->mpTextureParameterBindPoint[ii];
                SAFE_RELEASE((*pCur)->mppBindViews[bindPoint]);
                (*pCur)->mppBindViews[bindPoint] = ((CPUTTextureDX11*)(*pCur)->mpTexture[ii])->GetShaderResourceView();
                (*pCur)->mppBindViews[bindPoint]->AddRef();
            }
        }
        for( UINT ii=0; ii<(*pCur)->mBufferCount; ii++ )
        {
            if( (*pCur)->mpBuffer[ii] )
            {
                UINT bindPoint = (*pCur)->mpBufferParameterBindPoint[ii];
                SAFE_RELEASE((*pCur)->mppBindViews[bindPoint]);
                (*pCur)->mppBindViews[bindPoint] = ((CPUTBufferDX11*)(*pCur)->mpBuffer[ii])->GetShaderResourceView();
                (*pCur)->mppBindViews[bindPoint]->AddRef();
            }
        }
        for( UINT ii=0; ii<(*pCur)->mUAVCount; ii++ )
        {
            if( (*pCur)->mpUAV[ii] )
            {
                UINT bindPoint = (*pCur)->mpUAVParameterBindPoint[ii];
                SAFE_RELEASE((*pCur)->mppBindUAVs[bindPoint]);
                (*pCur)->mppBindUAVs[bindPoint] = ((CPUTBufferDX11*)(*pCur)->mpUAV[ii])->GetUnorderedAccessView();
			    (*pCur)->mppBindUAVs[bindPoint]->AddRef();
            }
        }
        for( UINT ii=0; ii<(*pCur)->mConstantBufferCount; ii++ )
        {
            if( (*pCur)->mpConstantBuffer[ii] )
            {
                UINT bindPoint = (*pCur)->mpConstantBufferParameterBindPoint[ii];
                SAFE_RELEASE((*pCur)->mppBindConstantBuffers[bindPoint]);
                (*pCur)->mppBindConstantBuffers[bindPoint] = ((CPUTBufferDX11*)(*pCur)->mpConstantBuffer[ii])->GetNativeBuffer();
                (*pCur)->mppBindConstantBuffers[bindPoint]->AddRef();
            }
        }
    }
}

//-----------------------------------------------------------------------------
void CPUTMaterialDX11::ReleaseTexturesAndBuffers()
{
    for( CPUTShaderParameters **pCur = mpShaderParametersList; *pCur; pCur++ ) // Release the buffers and views for each shader stage
    {
        if( *pCur )
        {
            for( UINT ii=0; ii<CPUT_MATERIAL_MAX_TEXTURE_SLOTS; ii++ )
            {
                SAFE_RELEASE((*pCur)->mppBindViews[ii]);
            }
            for( UINT ii=0; ii<CPUT_MATERIAL_MAX_UAV_SLOTS; ii++ )
            {
                SAFE_RELEASE((*pCur)->mppBindUAVs[ii]);
            }
            for( UINT ii=0; ii<CPUT_MATERIAL_MAX_CONSTANT_BUFFER_SLOTS; ii++ )
            {
                SAFE_RELEASE((*pCur)->mppBindConstantBuffers[ii]);
            }
        }
    }
}

//-----------------------------------------------------------------------------
void CPUTShaderParameters::CloneShaderParameters( CPUTShaderParameters *pShaderParameter )
{
    pShaderParameter->mpTextureParameterName              = new cString[mTextureParameterCount];
    pShaderParameter->mpTextureParameterBindPoint         = new UINT[   mTextureParameterCount];
    pShaderParameter->mpSamplerParameterName              = new cString[mSamplerParameterCount];
    pShaderParameter->mpSamplerParameterBindPoint         = new UINT[   mSamplerParameterCount];
    pShaderParameter->mpBufferParameterName               = new cString[mBufferParameterCount];
    pShaderParameter->mpBufferParameterBindPoint          = new UINT[   mBufferParameterCount];
    pShaderParameter->mpUAVParameterName                  = new cString[mUAVParameterCount];
    pShaderParameter->mpUAVParameterBindPoint             = new UINT[   mUAVParameterCount];
    pShaderParameter->mpConstantBufferParameterName       = new cString[mConstantBufferParameterCount];
    pShaderParameter->mpConstantBufferParameterBindPoint  = new UINT[   mConstantBufferParameterCount];

    pShaderParameter->mTextureCount                 = mTextureCount;
    pShaderParameter->mBufferCount                  = mBufferCount;
    pShaderParameter->mUAVCount                     = mUAVCount;
    pShaderParameter->mConstantBufferCount          = mConstantBufferCount;

    pShaderParameter->mTextureParameterCount        = mTextureParameterCount;
    pShaderParameter->mTextureParameterCount        = mTextureParameterCount;
    pShaderParameter->mBufferParameterCount         = mBufferParameterCount;
    pShaderParameter->mUAVParameterCount            = mUAVParameterCount;
    pShaderParameter->mConstantBufferParameterCount = mConstantBufferParameterCount;

    for(UINT ii=0; ii<mTextureParameterCount; ii++ )
    {
        pShaderParameter->mpTextureParameterName[ii]      = mpTextureParameterName[ii];
        pShaderParameter->mpTextureParameterBindPoint[ii] = mpTextureParameterBindPoint[ii];
    }
    for(UINT ii=0; ii<mSamplerParameterCount; ii++ )
    {
        pShaderParameter->mpSamplerParameterName[ii]      = mpSamplerParameterName[ii];
        pShaderParameter->mpSamplerParameterBindPoint[ii] = mpSamplerParameterBindPoint[ii];
    }
    for(UINT ii=0; ii<mBufferParameterCount; ii++ )
    {
        pShaderParameter->mpBufferParameterName[ii]      = mpBufferParameterName[ii];
        pShaderParameter->mpBufferParameterBindPoint[ii] = mpBufferParameterBindPoint[ii];
    }
    for(UINT ii=0; ii<mUAVParameterCount; ii++ )
    {
        pShaderParameter->mpUAVParameterName[ii]      = mpUAVParameterName[ii];
        pShaderParameter->mpUAVParameterBindPoint[ii] = mpUAVParameterBindPoint[ii];
    }
    for(UINT ii=0; ii<mConstantBufferParameterCount; ii++ )
    {
        pShaderParameter->mpConstantBufferParameterName[ii]      = mpConstantBufferParameterName[ii];
        pShaderParameter->mpConstantBufferParameterBindPoint[ii] = mpConstantBufferParameterBindPoint[ii];
    }
    pShaderParameter->mBindViewMin = mBindViewMin;
    pShaderParameter->mBindViewMax = mBindViewMax;

    pShaderParameter->mBindUAVMin = mBindUAVMin;
    pShaderParameter->mBindUAVMax = mBindUAVMax;

    pShaderParameter->mBindConstantBufferMin = mBindConstantBufferMin;
    pShaderParameter->mBindConstantBufferMax = mBindConstantBufferMax;
}