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
#include "CPUTAssetSetDX11.h"

#include "CPUTModelDX11.h"
#include "CPUTAssetLibraryDX11.h"
#include "CPUTCamera.h"
#include "CPUTLight.h"

//-----------------------------------------------------------------------------
CPUTAssetSetDX11::~CPUTAssetSetDX11()
{
    // Release all the elements in the asset list.  Note that we don't
    // recursively delete the hierarchy here.
    // We release the entries here because this class is where we add them.
    // TODO: Howevere, all derivations will have this, so perhaps it should go in the base.
    for( UINT ii=0; ii<mAssetCount; ii++ )
    {
        SAFE_RELEASE( mppAssetList[ii] );
    }
    SAFE_DELETE_ARRAY(mppAssetList);
}

//-----------------------------------------------------------------------------
CPUTResult CPUTAssetSetDX11::LoadAssetSet(cString name)
{
    CPUTResult result = CPUT_SUCCESS;

    // if not found, load the set file
    CPUTConfigFile ConfigFile;
    result = ConfigFile.LoadFile(name);
    if( !CPUTSUCCESS(result) )
    {
        return result;
    }
    // ASSERT( CPUTSUCCESS(result), _L("Failed loading set file '") + name + _L("'.") );

    mAssetCount = ConfigFile.BlockCount() + 1; // Add one for the implied root node
    mppAssetList = new CPUTRenderNode*[mAssetCount];
    mppAssetList[0] = mpRootNode;
    mpRootNode->AddRef();

    CPUTAssetLibraryDX11 *pAssetLibrary = (CPUTAssetLibraryDX11*)CPUTAssetLibrary::GetAssetLibrary();

    for(UINT ii=0; ii<mAssetCount-1; ii++) // Note: -1 because we added one for the root node (we don't load it)
    {
        CPUTConfigBlock *pBlock = ConfigFile.GetBlock(ii);
		ASSERT(pBlock != NULL, _L("Cannot find block"));
        cString nodeType = pBlock->GetValueByName(_L("type"))->ValueAsString();
        CPUTRenderNode *pParentNode = NULL;

        // TODO: use Get*() instead of Load*() ?

        cString name = pBlock->GetValueByName(_L("name"))->ValueAsString();

        int parentIndex;
        CPUTRenderNode *pNode = NULL;
        if(0==nodeType.compare(_L("null")))
        {
            pNode  = pNode = new CPUTNullNode();
            result = ((CPUTNullNode*)pNode)->LoadNullNode(pBlock, &parentIndex);
            pParentNode = mppAssetList[parentIndex+1];
            cString &parentPrefix = pParentNode->GetPrefix();
            pNode->SetPrefix( parentPrefix + _L(".") + name );
            pAssetLibrary->AddNullNode(parentPrefix + name, (CPUTNullNode*)pNode);
            // Add this null's name to our prefix
            // Append this null's name to our parent's prefix
            pNode->SetParent( pParentNode );
            pParentNode->AddChild( pNode );
        }
        else if(0==nodeType.compare(_L("model")))
        {
            CPUTConfigEntry *pValue = pBlock->GetValueByName( _L("instance") );
            CPUTModelDX11 *pModel = new CPUTModelDX11();
            if( pValue == &CPUTConfigEntry::sNullConfigValue )
            {
                // Not found.  So, not an instance.
                pModel->LoadModel(pBlock, &parentIndex, NULL);
            }
            else
            {
                int instance = pValue->ValueAsInt();
                pModel->LoadModel(pBlock, &parentIndex, (CPUTModel*)mppAssetList[instance+1]);
            }
            pParentNode = mppAssetList[parentIndex+1];
            pModel->SetParent( pParentNode );
            pParentNode->AddChild( pModel );
            cString &parentPrefix = pParentNode->GetPrefix();
            pModel->SetPrefix( parentPrefix );
            pAssetLibrary->AddModel(parentPrefix + name, pModel);

            pModel->UpdateBoundsWorldSpace();

#ifdef SUPPORT_DRAWING_BOUNDING_BOXES
            // Create a mesh for rendering the bounding box
            // TODO: There is definitely a better way to do this.  But, want to see the bounding boxes!
            pModel->CreateBoundingBoxMesh();
#endif
            pNode = pModel;
        }
        else if(0==nodeType.compare(_L("light")))
        {
            pNode = new CPUTLight();
            ((CPUTLight*)pNode)->LoadLight(pBlock, &parentIndex);
            pParentNode = mppAssetList[parentIndex+1]; // +1 because we added a root node to the start
            pNode->SetParent( pParentNode );
            pParentNode->AddChild( pNode );
            cString &parentPrefix = pParentNode->GetPrefix();
            pNode->SetPrefix( parentPrefix );
            pAssetLibrary->AddLight(parentPrefix + name, (CPUTLight*)pNode);
        }
        else if(0==nodeType.compare(_L("camera")))
        {
            pNode = new CPUTCamera();
            ((CPUTCamera*)pNode)->LoadCamera(pBlock, &parentIndex);
            pParentNode = mppAssetList[parentIndex+1]; // +1 because we added a root node to the start
            pNode->SetParent( pParentNode );
            pParentNode->AddChild( pNode );
            cString &parentPrefix = pParentNode->GetPrefix();
            pNode->SetPrefix( parentPrefix );
            pAssetLibrary->AddCamera(parentPrefix + name, (CPUTCamera*)pNode);
            if( !mpFirstCamera ) { mpFirstCamera = (CPUTCamera*)pNode; mpFirstCamera->AddRef();}
            ++mCameraCount;
        }
        else
        {
            ASSERT(0,_L("Unsupported node type '") + nodeType + _L("'."));
        }

        // Add the node to our asset list (i.e., the linear list, not the hierarchical)
        mppAssetList[ii+1] = pNode;
        // Don't AddRef.Creating it set the refcount to 1.  We add it to the list, and then we're done with it.
        // Net effect is 0 (+1 to add to list, and -1 because we're done with it)
        // pNode->AddRef();
    }
    return result;
}

//-----------------------------------------------------------------------------
CPUTAssetSet *CPUTAssetSetDX11::CreateAssetSet( const cString &name, const cString &absolutePathAndFilename )
{
    CPUTAssetLibraryDX11 *pAssetLibrary = ((CPUTAssetLibraryDX11*)CPUTAssetLibrary::GetAssetLibrary());

    // Create the root node.
    CPUTNullNode *pRootNode = new CPUTNullNode();
    pRootNode->SetName(_L("_CPUTAssetSetRootNode_"));

    // Create the asset set, set its root, and load it
    CPUTAssetSet   *pNewAssetSet = new CPUTAssetSetDX11();
    pNewAssetSet->SetRoot( pRootNode );
    pAssetLibrary->AddNullNode( name + _L("_Root"), pRootNode );

    CPUTResult result = pNewAssetSet->LoadAssetSet(absolutePathAndFilename);
    if( CPUTSUCCESS(result) )
    {
        pAssetLibrary->AddAssetSet(name, pNewAssetSet);
        return pNewAssetSet;
    }
    ASSERT( CPUTSUCCESS(result), _L("Error loading AssetSet\n'")+absolutePathAndFilename+_L("'"));
    pNewAssetSet->Release();
    return NULL;
}

