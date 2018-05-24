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
#ifndef __CPUTASSETLIBRARY_H__
#define __CPUTASSETLIBRARY_H__

#include "CPUT.h"
#include "CPUTOSServicesWin.h" // TODO: why is this windows-specific?

// Global Asset Library
//
// The purpose of this library is to keep a copy of all loaded assets and
// provide a one-stop-loading system.  All assets that are loaded into the
// system via the Getxxx() operators stays in the library.  Further Getxxx()
// operations on an already loaded object will addref and return the previously
// loaded object.
//-----------------------------------------------------------------------------
// node that holds a single library object
struct CPUTAssetListEntry
{
    UINT                hash;
    cString             name;
    void               *pData;
    CPUTAssetListEntry *pNext;
};
#define SAFE_RELEASE_LIST(list) ReleaseList(list);(list)=NULL;

class CPUTAssetSet;
class CPUTNullNode;
class CPUTModel;
class CPUTMaterial;
class CPUTLight;
class CPUTCamera;
class CPUTTexture;
class CPUTBuffer;
class CPUTRenderStateBlock;
class CPUTFont;

class CPUTAssetLibrary
{
protected:
    static CPUTAssetLibrary *mpAssetLibrary;

    // simple linked lists for now, but if we want to optimize or load blocks
    // we can change these to dynamically re-sizing arrays and then just do
    // memcopies into the structs.
    // Note: No camera, light, or NullNode directory names - they don't have payload files (i.e., they are defined completely in the .set file)
    cString  mAssetSetDirectoryName;
    cString  mModelDirectoryName;
    cString  mMaterialDirectoryName;
    cString  mTextureDirectoryName;
    cString  mShaderDirectoryName;
    cString  mFontDirectoryName;

public: // TODO: temporary for debug.
    // TODO: Make these lists static.  Share assets (e.g., texture) across all requests for this process.
    static CPUTAssetListEntry  *mpAssetSetList;
    static CPUTAssetListEntry  *mpNullNodeList;
    static CPUTAssetListEntry  *mpModelList;
    static CPUTAssetListEntry  *mpCameraList;
    static CPUTAssetListEntry  *mpLightList;
    static CPUTAssetListEntry  *mpMaterialList;
    static CPUTAssetListEntry  *mpTextureList;
    static CPUTAssetListEntry  *mpBufferList;
    static CPUTAssetListEntry  *mpConstantBufferList;
    static CPUTAssetListEntry  *mpRenderStateBlockList;
    static CPUTAssetListEntry  *mpFontList;

    static void RebindTexturesAndBuffers();
    static void ReleaseTexturesAndBuffers();

public:
    static CPUTAssetLibrary *GetAssetLibrary(){ return mpAssetLibrary; }
    static void              DeleteAssetLibrary();

    CPUTAssetLibrary() {}
    virtual ~CPUTAssetLibrary() {}

    // Add/get/delete items to specified library
    void *FindAsset(const cString &name, CPUTAssetListEntry *pList, bool nameIsFullPathAndFilename=false);
    virtual void ReleaseAllLibraryLists();

    void SetMediaDirectoryName( const cString &directoryName)
    {
        mAssetSetDirectoryName = directoryName + _L("Asset\\");
        mModelDirectoryName    = directoryName + _L("Asset\\");
        mMaterialDirectoryName = directoryName + _L("Material\\");
        mTextureDirectoryName  = directoryName + _L("Texture\\");
        mShaderDirectoryName   = directoryName + _L("Shader\\");
        // mFontStateBlockDirectoryName   = directoryName + _L("Font\\");
    }
    void SetAssetSetDirectoryName(        const cString &directoryName) { mAssetSetDirectoryName  = directoryName; }
    void SetModelDirectoryName(           const cString &directoryName) { mModelDirectoryName     = directoryName; }
    void SetMaterialDirectoryName(        const cString &directoryName) { mMaterialDirectoryName  = directoryName; }
    void SetTextureDirectoryName(         const cString &directoryName) { mTextureDirectoryName   = directoryName; }
    void SetShaderDirectoryName(          const cString &directoryName) { mShaderDirectoryName    = directoryName; }
    void SetFontDirectoryName(            const cString &directoryName) { mFontDirectoryName      = directoryName; }
    void SetAllAssetDirectoryNames(       const cString &directoryName) {
            mAssetSetDirectoryName       = directoryName;
            mModelDirectoryName          = directoryName;
            mMaterialDirectoryName       = directoryName;
            mTextureDirectoryName        = directoryName;
            mShaderDirectoryName         = directoryName;
            mFontDirectoryName           = directoryName;
    };

    cString &GetAssetSetDirectoryName() { return mAssetSetDirectoryName; }
    cString &GetModelDirectory()        { return mModelDirectoryName; }
    cString &GetMaterialDirectory()     { return mMaterialDirectoryName; }
    cString &GetTextureDirectory()      { return mTextureDirectoryName; }
    cString &GetShaderDirectory()       { return mShaderDirectoryName; }
    cString &GetFontDirectory()         { return mFontDirectoryName; }

    void AddAssetSet(        const cString &name, CPUTAssetSet         *pAssetSet)        { AddAsset( name, pAssetSet,         &mpAssetSetList ); }
    void AddNullNode(        const cString &name, CPUTNullNode         *pNullNode)        { AddAsset( name, pNullNode,         &mpNullNodeList ); }
    void AddModel(           const cString &name, CPUTModel            *pModel)           { AddAsset( name, pModel,            &mpModelList ); }
    void AddMaterial(        const cString &name, CPUTMaterial         *pMaterial)        { AddAsset( name, pMaterial,         &mpMaterialList ); }
    void AddLight(           const cString &name, CPUTLight            *pLight)           { AddAsset( name, pLight,            &mpLightList ); }
    void AddCamera(          const cString &name, CPUTCamera           *pCamera)          { AddAsset( name, pCamera,           &mpCameraList ); }
    void AddTexture(         const cString &name, CPUTTexture          *pTexture)         { AddAsset( name, pTexture,          &mpTextureList ); }
    void AddBuffer(          const cString &name, CPUTBuffer           *pBuffer)          { AddAsset( name, pBuffer,           &mpBufferList ); }
    void AddConstantBuffer(  const cString &name, CPUTBuffer           *pBuffer)          { AddAsset( name, pBuffer,           &mpConstantBufferList ); }
    void AddRenderStateBlock(const cString &name, CPUTRenderStateBlock *pRenderStateBlock){ AddAsset( name, pRenderStateBlock, &mpRenderStateBlockList ); }
    void AddFont(            const cString &name, CPUTFont             *pFont)            { AddAsset( name, pFont,             &mpFontList ); }

    CPUTAssetSet *FindAssetSet(const cString &name, bool nameIsFullPathAndFilename=false)       { return (CPUTAssetSet*)FindAsset( name, mpAssetSetList, nameIsFullPathAndFilename ); }
    CPUTNullNode *FindNullNode(const cString &name, bool nameIsFullPathAndFilename=false)       { return (CPUTNullNode*)FindAsset( name, mpNullNodeList, nameIsFullPathAndFilename ); }
    CPUTModel    *FindModel(const cString &name, bool nameIsFullPathAndFilename=false)          { return (CPUTModel*)FindAsset(    name, mpModelList,    nameIsFullPathAndFilename ); }
    CPUTMaterial *FindMaterial(const cString &name, bool nameIsFullPathAndFilename=false)       { return (CPUTMaterial*)FindAsset( name, mpMaterialList, nameIsFullPathAndFilename ); }
    CPUTLight    *FindLight(const cString &name, bool nameIsFullPathAndFilename=false)          { return (CPUTLight*)FindAsset(    name, mpLightList,    nameIsFullPathAndFilename ); }
    CPUTCamera   *FindCamera(const cString &name, bool nameIsFullPathAndFilename=false)         { return (CPUTCamera*)FindAsset(   name, mpCameraList,   nameIsFullPathAndFilename ); }
    CPUTTexture  *FindTexture(const cString &name, bool nameIsFullPathAndFilename=false)        { return (CPUTTexture*)FindAsset(  name, mpTextureList,  nameIsFullPathAndFilename ); }
    CPUTBuffer   *FindBuffer(const cString &name, bool nameIsFullPathAndFilename=false)         { return (CPUTBuffer*)FindAsset(   name, mpBufferList,   nameIsFullPathAndFilename ); }
    CPUTBuffer   *FindConstantBuffer(const cString &name, bool nameIsFullPathAndFilename=false) { return (CPUTBuffer*)FindAsset(   name, mpConstantBufferList, nameIsFullPathAndFilename ); }
    CPUTRenderStateBlock *FindRenderStateBlock(const cString &name, bool nameIsFullPathAndFilename=false ) { return (CPUTRenderStateBlock*)FindAsset( name, mpRenderStateBlockList, nameIsFullPathAndFilename ); }
    CPUTFont     *FindFont(const cString &name, bool nameIsFullPathAndFilename=false)           { return (CPUTFont*)FindAsset(     name, mpFontList,     nameIsFullPathAndFilename ); }

    // If the asset exists, these 'Get' methods will addref and return it.  Otherwise,
    // they will create it and return it.
    CPUTAssetSet         *GetAssetSet(        const cString &name, bool nameIsFullPathAndFilename=false );
    CPUTTexture          *GetTexture(         const cString &name, bool nameIsFullPathAndFilename=false, bool loadAsSRGB=true );
    CPUTMaterial         *GetMaterial(        const cString &name, bool nameIsFullPathAndFilename=false, const cString &modelSuffix=_L(""), const cString &meshSuffix=_L("") );
    CPUTModel            *GetModel(           const cString &name, bool nameIsFullPathAndFilename=false  );
    CPUTRenderStateBlock *GetRenderStateBlock(const cString &name, bool nameIsFullPathAndFilename=false);
    CPUTBuffer           *GetBuffer(          const cString &name );
    CPUTBuffer           *GetConstantBuffer(  const cString &name );
    CPUTCamera           *GetCamera(          const cString &name );
    CPUTLight            *GetLight(           const cString &name );
    CPUTFont             *GetFont(            const cString &name );

protected:
    // helper functions
    void ReleaseList(CPUTAssetListEntry *pLibraryRoot);
    void AddAsset( const cString &name, void *pAsset, CPUTAssetListEntry **pHead );
    UINT CPUTComputeHash( const cString &string )
    {
        size_t length = string.length();
        UINT hash = 0;
        for( size_t ii=0; ii<length; ii++ )
        {
            hash += tolower(string[ii]);
        }
        return hash;
    }
};

#endif //#ifndef __CPUTASSETLIBRARY_H__


