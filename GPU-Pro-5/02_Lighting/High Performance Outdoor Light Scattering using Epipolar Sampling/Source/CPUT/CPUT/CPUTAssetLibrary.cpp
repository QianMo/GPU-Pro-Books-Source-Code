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
#include <algorithm> // for std::string.transform()
#include "CPUTAssetLibrary.h"
#include "CPUTRenderNode.h"
#include "CPUTAssetSet.h"
#include "CPUTMaterial.h"
#include "CPUTRenderStateBlock.h"
#include "CPUTModel.h"
#include "CPUTCamera.h"
#include "CPUTLight.h"
#include "CPUTFont.h"
#include "CPUTBuffer.h"

// TODO: How do we want to store both DX and OGL assets in the same library?
//       Could use API-specific libraries.  Could move both platforms to single file.  etc.
// CPUTAssetLibrary   *CPUTAssetLibrary::mpAssetLibrary = new CPUTAssetLibraryDX11();
CPUTAssetListEntry *CPUTAssetLibrary::mpAssetSetList = NULL;
CPUTAssetListEntry *CPUTAssetLibrary::mpNullNodeList = NULL;
CPUTAssetListEntry *CPUTAssetLibrary::mpModelList = NULL;
CPUTAssetListEntry *CPUTAssetLibrary::mpCameraList = NULL;
CPUTAssetListEntry *CPUTAssetLibrary::mpLightList = NULL;
CPUTAssetListEntry *CPUTAssetLibrary::mpMaterialList = NULL;
CPUTAssetListEntry *CPUTAssetLibrary::mpTextureList = NULL;
CPUTAssetListEntry *CPUTAssetLibrary::mpBufferList = NULL;
CPUTAssetListEntry *CPUTAssetLibrary::mpConstantBufferList = NULL;
CPUTAssetListEntry *CPUTAssetLibrary::mpRenderStateBlockList = NULL;
CPUTAssetListEntry *CPUTAssetLibrary::mpFontList = NULL;

//-----------------------------------------------------------------------------
void CPUTAssetLibrary::ReleaseTexturesAndBuffers()
{
    CPUTAssetListEntry *pMaterial = mpMaterialList;
    while( pMaterial )
    {
        ((CPUTMaterial*)pMaterial->pData)->ReleaseTexturesAndBuffers();
        pMaterial = pMaterial->pNext;
    }
}

//-----------------------------------------------------------------------------
void CPUTAssetLibrary::RebindTexturesAndBuffers()
{
    CPUTAssetListEntry *pMaterial = mpMaterialList;
    while( pMaterial )
    {
        ((CPUTMaterial*)pMaterial->pData)->RebindTexturesAndBuffers();
        pMaterial = pMaterial->pNext;
    }
}

//-----------------------------------------------------------------------------
void CPUTAssetLibrary::DeleteAssetLibrary()
{
    SAFE_DELETE(mpAssetLibrary);
}

//-----------------------------------------------------------------------------
void CPUTAssetLibrary::ReleaseAllLibraryLists()
{
    // Release philosophy:  Everyone that references releases.  If node refers to parent, then it should release parent, etc...
    // TODO: Traverse lists.  Print names and ref counts (as debug aid)
#undef SAFE_RELEASE_LIST
#define SAFE_RELEASE_LIST(x) {ReleaseList(x); x = NULL;}

    SAFE_RELEASE_LIST(mpAssetSetList);
    SAFE_RELEASE_LIST(mpMaterialList);
    SAFE_RELEASE_LIST(mpModelList);
    SAFE_RELEASE_LIST(mpLightList);
    SAFE_RELEASE_LIST(mpCameraList);
    SAFE_RELEASE_LIST(mpNullNodeList);
    SAFE_RELEASE_LIST(mpTextureList );
    SAFE_RELEASE_LIST(mpBufferList );
    SAFE_RELEASE_LIST(mpConstantBufferList );
    SAFE_RELEASE_LIST(mpRenderStateBlockList );
    SAFE_RELEASE_LIST(mpFontList);

    // The following -specific items are destroyed in the derived class
    // TODO.  Move their declaration and definition to the derived class too
    // SAFE_RELEASE_LIST(mpPixelShaderList);
    // SAFE_RELEASE_LIST(mpVertexShaderList);
    // SAFE_RELEASE_LIST(mpGeometryShaderList);
}

//-----------------------------------------------------------------------------
void CPUTAssetLibrary::ReleaseList(CPUTAssetListEntry *pLibraryRoot)
{
    CPUTAssetListEntry *pNext;
    for( CPUTAssetListEntry *pNodeEntry = pLibraryRoot; NULL != pNodeEntry; pNodeEntry = pNext )
    {
        pNext = pNodeEntry->pNext;
        CPUTRefCount *pRefCountedNode = (CPUTRefCount*)pNodeEntry->pData;
        pRefCountedNode->Release();
        HEAPCHECK;
        delete pNodeEntry;
    }
}

// Find an asset in a specific library
// ** Does not Addref() returned items **
// Asset library doesn't care if we're using absolute paths for names or not, it
// just adds/finds/deletes the matching string literal.
//-----------------------------------------------------------------------------
void *CPUTAssetLibrary::FindAsset(const cString &name, CPUTAssetListEntry *pList, bool nameIsFullPathAndFilename)
{
    cString absolutePathAndFilename;
    CPUTOSServices *pServices = CPUTOSServices::GetOSServices();
    pServices->ResolveAbsolutePathAndFilename( nameIsFullPathAndFilename ? name : (mAssetSetDirectoryName + name), &absolutePathAndFilename);
    absolutePathAndFilename = nameIsFullPathAndFilename ? name : absolutePathAndFilename;

    UINT hash = CPUTComputeHash( absolutePathAndFilename );
    while(NULL!=pList)
    {
        if( hash == pList->hash && (0 == _wcsicmp( absolutePathAndFilename.data(), pList->name.data() )) )
        {
            return (void*)pList->pData;
        }
        pList = pList->pNext;
    }
    return NULL;
}

//-----------------------------------------------------------------------------
void CPUTAssetLibrary::AddAsset(const cString &name, void *pAsset, CPUTAssetListEntry **pHead)
{
    // convert string to lowercase
    cString lowercaseName = name;
    std::transform(lowercaseName.begin(), lowercaseName.end(), lowercaseName.begin(), ::tolower);

    // Do we already have one by this name?
    CPUTAssetListEntry *pTail = *pHead;
    // TODO:  Save explicit tail pointer instead of iterating to find the null.
    for( CPUTAssetListEntry *pCur=*pHead; NULL!=pCur; pCur=pCur->pNext )
    {
        // Assert that we haven't added one with this name
        ASSERT( 0 != _wcsicmp( pCur->name.data(), name.data() ), _L("Warning: asset ")+name+_L(" already exists") );
        pTail = pCur;
    }
    CPUTAssetListEntry **pDest = pTail ? &pTail->pNext : pHead;
    *pDest = new CPUTAssetListEntry();
    (*pDest)->hash = CPUTComputeHash(name);
    (*pDest)->name = name;
    (*pDest)->pData = pAsset;
    (*pDest)->pNext = NULL;

    // TODO: Our assets are not yet all derived from CPUTRenderNode.
    // TODO: For now, rely on caller performing the AddRef() as it knows the assets type.
    ((CPUTRefCount*)pAsset)->AddRef();
}

//-----------------------------------------------------------------------------
CPUTRenderStateBlock *CPUTAssetLibrary::GetRenderStateBlock(const cString &name, bool nameIsFullPathAndFilename )
{
    // Resolve name to absolute path before searching
    cString finalName;
    if( name.at(0) == '$' )
    {
        finalName = name;
    } else
    {
        // Resolve name to absolute path
        CPUTOSServices *pServices = CPUTOSServices::GetOSServices();
        pServices->ResolveAbsolutePathAndFilename( nameIsFullPathAndFilename? name : (mShaderDirectoryName + name), &finalName);
    }

    // see if the render state block is already in the library
    CPUTRenderStateBlock *pRenderStateBlock = FindRenderStateBlock(finalName, true);
    if(NULL==pRenderStateBlock)
    {
        return CPUTRenderStateBlock::CreateRenderStateBlock( name, finalName );
    }
    pRenderStateBlock->AddRef();
    return pRenderStateBlock;
}

//-----------------------------------------------------------------------------
CPUTAssetSet *CPUTAssetLibrary::GetAssetSet( const cString &name, bool nameIsFullPathAndFilename )
{
    // Resolve the absolute path
    cString absolutePathAndFilename;
    CPUTOSServices *pServices = CPUTOSServices::GetOSServices();
    pServices->ResolveAbsolutePathAndFilename( nameIsFullPathAndFilename ? name
        : (mAssetSetDirectoryName + name + _L(".set")), &absolutePathAndFilename );
    absolutePathAndFilename = nameIsFullPathAndFilename ? name : absolutePathAndFilename;

    CPUTAssetSet *pAssetSet = FindAssetSet(absolutePathAndFilename, true);
    if(NULL == pAssetSet)
    {
        return CPUTAssetSet::CreateAssetSet( name, absolutePathAndFilename );
    }
    pAssetSet->AddRef();
    return pAssetSet;
}


// TODO: All of these Get() functions look very similar.
// Keep them all for their interface, but have them call a common function
//-----------------------------------------------------------------------------
CPUTMaterial *CPUTAssetLibrary::GetMaterial(const cString &name, bool nameIsFullPathAndFilename, const cString &modelSuffix, const cString &meshSuffix)
{
    // Resolve name to absolute path before searching
    CPUTOSServices *pServices = CPUTOSServices::GetOSServices();
    cString absolutePathAndFilename;
    pServices->ResolveAbsolutePathAndFilename( nameIsFullPathAndFilename? name : (mMaterialDirectoryName + name + _L(".mtl")), &absolutePathAndFilename);

    // If we already have one by this name, then return it
    CPUTMaterial *pMaterial = FindMaterial(absolutePathAndFilename, true);
    if(NULL==pMaterial)
    {
        // We don't already have it in the library, so create it.
        pMaterial = CPUTMaterial::CreateMaterial( absolutePathAndFilename, modelSuffix, meshSuffix );
        return pMaterial;
    }
    else if( (0==modelSuffix.length()) && !pMaterial->MaterialRequiresPerModelPayload() )
    {
        // This material doesn't have per-model elements, so we don't need to clone it.
        pMaterial->AddRef();
        return pMaterial;
    }

#ifdef _DEBUG
    // We need to clone the material.  Do that by loading it again, but with a different name.
    // Add the model's suffix (address as string, plus model's material array index as string)
    CPUTMaterial *pUniqueMaterial = FindMaterial(absolutePathAndFilename + modelSuffix + meshSuffix, true);
    ASSERT( NULL == pUniqueMaterial, _L("Unique material already not unique: ") + absolutePathAndFilename + modelSuffix + meshSuffix );
#endif

    CPUTMaterial *pClonedMaterial = pMaterial->CloneMaterial( absolutePathAndFilename, modelSuffix, meshSuffix );
    AddMaterial( absolutePathAndFilename + modelSuffix + meshSuffix, pClonedMaterial );

    return pClonedMaterial;
}

// Get CPUTModel from asset library
// If the model exists, then the existing model is Addref'ed and returned
//-----------------------------------------------------------------------------
CPUTModel *CPUTAssetLibrary::GetModel(const cString &name, bool nameIsFullPathAndFilename)
{
    // Resolve name to absolute path before searching
    cString absolutePathAndFilename;
    CPUTOSServices *pServices = CPUTOSServices::GetOSServices();
    pServices->ResolveAbsolutePathAndFilename( nameIsFullPathAndFilename? name : (mModelDirectoryName + name + _L(".mdl")), &absolutePathAndFilename);
    absolutePathAndFilename = nameIsFullPathAndFilename ? name : absolutePathAndFilename;

    // If we already have one by this name, then return it
    CPUTModel *pModel = FindModel(absolutePathAndFilename, true);
    if(NULL!=pModel)
    {
        pModel->AddRef();
        return pModel;
    }

    // Looks like no one calls GetModel().  Or, they never call it for missing models.
#if TODO // delegate
    // Model was not in the library, so create and load a new model
    pModel = new CPUTModel();
    pModel->LoadModelPayload(absolutePathAndFilename);
    AddModel(name, pModel);

    return CPUTModel::CreateMode( absolutePathAndFilename, aboslutePathAndFilename );
#endif

    return pModel;
}

//-----------------------------------------------------------------------------
CPUTCamera *CPUTAssetLibrary::GetCamera(const cString &name)
{
    // TODO: Should we prefix camera names with a path anyway?  To keek them unique?
    // If we already have one by this name, then return it
    CPUTCamera *pCamera = FindCamera(name, true);
    if(NULL!=pCamera)
    {
        pCamera->AddRef();
        return pCamera;
    }
    return NULL;
}

//-----------------------------------------------------------------------------
CPUTLight *CPUTAssetLibrary::GetLight(const cString &name)
{
    // If we already have one by this name, then return it
    CPUTLight *pLight = FindLight(name, true);
    if(NULL!=pLight)
    {
        pLight->AddRef();
        return pLight;
    }
    return NULL;
}

//-----------------------------------------------------------------------------
CPUTTexture *CPUTAssetLibrary::GetTexture(const cString &name, bool nameIsFullPathAndFilename, bool loadAsSRGB )
{
    cString finalName;
    if( name.at(0) == '$' )
    {
        finalName = name;
    } else
    {
        // Resolve name to absolute path
        CPUTOSServices *pServices = CPUTOSServices::GetOSServices();
        pServices->ResolveAbsolutePathAndFilename( nameIsFullPathAndFilename? name : (mTextureDirectoryName + name), &finalName);
    }
    // If we already have one by this name, then return it
    CPUTTexture *pTexture = FindTexture(finalName, true);
    if(NULL==pTexture)
    {
        return CPUTTexture::CreateTexture( name, finalName, loadAsSRGB);
    }
    pTexture->AddRef();
    return pTexture;
}

//-----------------------------------------------------------------------------
CPUTBuffer *CPUTAssetLibrary::GetBuffer(const cString &name )
{
    // If we already have one by this name, then return it
    CPUTBuffer *pBuffer = FindBuffer(name, true);
    ASSERT( pBuffer, _L("Can't find buffer ") + name );
    pBuffer->AddRef();
    return pBuffer;
}

//-----------------------------------------------------------------------------
CPUTBuffer *CPUTAssetLibrary::GetConstantBuffer(const cString &name )
{
    // If we already have one by this name, then return it
    CPUTBuffer *pBuffer = FindConstantBuffer(name, true);
    ASSERT( pBuffer, _L("Can't find constant buffer ") + name );
    pBuffer->AddRef();
    return pBuffer;
}

//-----------------------------------------------------------------------------
CPUTFont *CPUTAssetLibrary::GetFont(const cString &name )
{
    // Resolve name to absolute path
    CPUTOSServices *pServices = CPUTOSServices::GetOSServices();
    cString absolutePathAndFilename;
    pServices->ResolveAbsolutePathAndFilename( (mFontDirectoryName + name), &absolutePathAndFilename);

    // If we already have one by this name, then return it
    CPUTFont *pFont = FindFont(absolutePathAndFilename, true);
    if(NULL==pFont)
    {
        return CPUTFont::CreateFont( name, absolutePathAndFilename);
    }
    pFont->AddRef();
    return pFont;
}
