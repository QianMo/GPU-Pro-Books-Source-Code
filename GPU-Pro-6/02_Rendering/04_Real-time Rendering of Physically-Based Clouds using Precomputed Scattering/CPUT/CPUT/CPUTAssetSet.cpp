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
#include "CPUTAssetSet.h"
#ifdef CPUT_FOR_DX11
    #include "CPUTAssetLibraryDX11.h"
#else    
    #error You must supply a target graphics API (ex: #define CPUT_FOR_DX11), or implement the target API for this file.
#endif


//-----------------------------------------------------------------------------
CPUTAssetSet::CPUTAssetSet() :
    mppAssetList(NULL),
    mAssetCount(0),
    mpRootNode(NULL),
    mpFirstCamera(NULL),
    mCameraCount(0)
{
}

//-----------------------------------------------------------------------------
CPUTAssetSet::~CPUTAssetSet()
{
    SAFE_RELEASE(mpFirstCamera);

    // Deleteing the asset set implies recursively releasing all the assets in the hierarchy
    if(mpRootNode && !mpRootNode->ReleaseRecursive() )
    {
        mpRootNode = NULL;
    }
}

//-----------------------------------------------------------------------------
void CPUTAssetSet::RenderRecursive(CPUTRenderParameters &renderParams )
{
    if(mpRootNode)
    {
        mpRootNode->RenderRecursive(renderParams);
    }
}

//-----------------------------------------------------------------------------
void CPUTAssetSet::RenderShadowRecursive(CPUTRenderParameters &renderParams )
{
    if(mpRootNode)
    {
        mpRootNode->RenderShadowRecursive(renderParams);
    }
}

//-----------------------------------------------------------------------------
void CPUTAssetSet::GetBoundingBox(float3 *pCenter, float3 *pHalf)
{
    *pCenter = *pHalf = float3(0.0f);
    if(mpRootNode)
    {
        mpRootNode->GetBoundingBoxRecursive(pCenter, pHalf);
    }
}

//-----------------------------------------------------------------------------
CPUTResult CPUTAssetSet::GetAssetByIndex(const UINT index, CPUTRenderNode **ppRenderNode)
{
    ASSERT( NULL != ppRenderNode, _L("Invalid NULL parameter") );
    *ppRenderNode = mppAssetList[index];
    mppAssetList[index]->AddRef();
    return CPUT_SUCCESS;
}

// Note: We create an object of derived type here.  What do we want to do when we also support OGL?  Have DX and OGL specific Create functions?
#include "CPUTAssetSetDX11.h"
//-----------------------------------------------------------------------------
CPUTAssetSet *CPUTAssetSet::CreateAssetSet( const cString &name, const cString &absolutePathAndFilename )
{
    // TODO: accept DX11/OGL param to control which platform we generate.
    // TODO: be sure to support the case where we want to support only one of them
#ifdef CPUT_FOR_DX11
    return CPUTAssetSetDX11::CreateAssetSet( name, absolutePathAndFilename );
#else    
    #error You must supply a target graphics API (ex: #define CPUT_FOR_DX11), or implement the target API for this file.
#endif
    
}

