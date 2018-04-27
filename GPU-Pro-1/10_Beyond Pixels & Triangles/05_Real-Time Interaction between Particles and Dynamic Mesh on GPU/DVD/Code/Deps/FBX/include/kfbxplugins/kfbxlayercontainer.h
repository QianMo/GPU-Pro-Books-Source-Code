/*!  \file kfbxlayercontainer.h
 */

#ifndef _FBXSDK_LAYER_CONTAINER_H_
#define _FBXSDK_LAYER_CONTAINER_H_

/**************************************************************************************

 Copyright © 2001 - 2008 Autodesk, Inc. and/or its licensors.
 All Rights Reserved.

 The coded instructions, statements, computer programs, and/or related material 
 (collectively the "Data") in these files contain unpublished information 
 proprietary to Autodesk, Inc. and/or its licensors, which is protected by 
 Canada and United States of America federal copyright law and by international 
 treaties. 
 
 The Data may not be disclosed or distributed to third parties, in whole or in
 part, without the prior written consent of Autodesk, Inc. ("Autodesk").

 THE DATA IS PROVIDED "AS IS" AND WITHOUT WARRANTY.
 ALL WARRANTIES ARE EXPRESSLY EXCLUDED AND DISCLAIMED. AUTODESK MAKES NO
 WARRANTY OF ANY KIND WITH RESPECT TO THE DATA, EXPRESS, IMPLIED OR ARISING
 BY CUSTOM OR TRADE USAGE, AND DISCLAIMS ANY IMPLIED WARRANTIES OF TITLE, 
 NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE OR USE. 
 WITHOUT LIMITING THE FOREGOING, AUTODESK DOES NOT WARRANT THAT THE OPERATION
 OF THE DATA WILL BE UNINTERRUPTED OR ERROR FREE. 
 
 IN NO EVENT SHALL AUTODESK, ITS AFFILIATES, PARENT COMPANIES, LICENSORS
 OR SUPPLIERS ("AUTODESK GROUP") BE LIABLE FOR ANY LOSSES, DAMAGES OR EXPENSES
 OF ANY KIND (INCLUDING WITHOUT LIMITATION PUNITIVE OR MULTIPLE DAMAGES OR OTHER
 SPECIAL, DIRECT, INDIRECT, EXEMPLARY, INCIDENTAL, LOSS OF PROFITS, REVENUE
 OR DATA, COST OF COVER OR CONSEQUENTIAL LOSSES OR DAMAGES OF ANY KIND),
 HOWEVER CAUSED, AND REGARDLESS OF THE THEORY OF LIABILITY, WHETHER DERIVED
 FROM CONTRACT, TORT (INCLUDING, BUT NOT LIMITED TO, NEGLIGENCE), OR OTHERWISE,
 ARISING OUT OF OR RELATING TO THE DATA OR ITS USE OR ANY OTHER PERFORMANCE,
 WHETHER OR NOT AUTODESK HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH LOSS
 OR DAMAGE. 

**************************************************************************************/

#include <kaydaradef.h>
#ifndef KFBX_DLL 
	#define KFBX_DLL K_DLLIMPORT
#endif

#include <kaydara.h>

#include <kfbxplugins/kfbxnodeattribute.h>
#include <kfbxplugins/kfbxlayer.h>

#include <klib/karrayul.h>

#include <kfbxmath/kfbxvector4.h>

#include <fbxfilesdk_nsbegin.h>

class KFbxSdkManager;


/** \brief KFbxLayerContainer is the base class for managing Layers. 
  * This class manages the creation and destruction of layers. 
  * A Layer contains Layer Element(s) of the following types: 
  *      \li Normals
  *      \li Materials
  *      \li Polygon Groups
  *      \li UVs
  *      \li Vertex Color
  *      \li Textures
  * See KFbxLayerElement for more details.
  * \nosubgrouping
  */
class KFBX_DLL KFbxLayerContainer : public KFbxNodeAttribute
{
	KFBXOBJECT_DECLARE(KFbxLayerContainer,KFbxNodeAttribute);
public:

	/** Return the type of node attribute.
	  * This class is pure virtual.
	  */
	virtual EAttributeType GetAttributeType() const { return eUNIDENTIFIED; } 

	/**
	  * \name Layer Management 
	  */
	//@{

	/** Create a new layer on top of existing layers.
	  * \return     Index of created layer or -1 if an error occured.
	  */
	int CreateLayer();

	//! Delete all layers.
    void ClearLayers();

	/** Get number of layers.
	 * \return     Return the number of layers.
	 */
	int GetLayerCount() const;

	/** Get number of layers containing the specified layer element type.
	  * \param pType     The requested Layer Element type.
      * \param pUVCount  When \c true, request the number of UV layers connected to the specified Layer Element type.
	  * \return          The number of layers containing a layer of type pType.
	  */
	int GetLayerCount(KFbxLayerElement::ELayerElementType pType,  bool pUVCount=false) const;

	/** Get the layer at given index.
	  *	\param pIndex     Layer index.
	  * \return           Pointer to the layer, or \c NULL if pIndex is out of range.
	  */
	KFbxLayer* GetLayer(int pIndex);

	/** Get the layer at given index.
	  *	\param pIndex     Layer index.
	  * \return           Pointer to the layer, or \c NULL if pIndex is out of range.
	  */
	KFbxLayer const* GetLayer(int pIndex) const;

	/** Get the n'th layer containing the specified layer element type.
	  *	\param pIndex     Layer index.
	  * \param pType      Layer element type.
      * \param pIsUV      When \c true, request the UV LayerElement connected to the specified Layer Element type.
	  * \return           Pointer to the layer, or \c NULL if pIndex is out of range for the specified type (pType).
	  */
	KFbxLayer* GetLayer(int pIndex, KFbxLayerElement::ELayerElementType pType, bool pIsUV=false);

	/** Get the n'th layer containing the specified layer element type.
	  *	\param pIndex     Layer index.
	  * \param pType      Layer element type.
      * \param pIsUV      When \c true request the UV LayerElement connected to the specified Layer Element type.
	  * \return           Pointer to the layer, or \c NULL if pIndex is out of range for the specified type (pType).
	  */
	KFbxLayer const* GetLayer(int pIndex, KFbxLayerElement::ELayerElementType pType, bool pIsUV=false) const;

	/**	Get the index of n'th layer containing the specified layer element type.
	  * \param pIndex     Layer index of the specified type.
	  * \param pType      Layer type.
      * \param pIsUV      When \c true request the index of the UV LayerElement connected to the specified Layer Element type.
	  * \return           Index of the specified layer type, or -1 if the layer is not found.
	  * \remarks          The returned index is the position of the layer in the global array of layers.
	  *                   You can use the returned index to call GetLayer(int pIndex).
	  */
	int GetLayerIndex(int pIndex, KFbxLayerElement::ELayerElementType pType, bool pIsUV=false) const;

	/** Convert the global index of the layer to a type-specific index.
	  * \param pGlobalIndex     The index of the layer in the global array of layers.
	  * \param pType            The type uppon which the typed index will be returned.
      * \param pIsUV            When \c true request the index of the UV LayerElement connected to the specified Layer Element type.
	  * \return                 Index of the requested layer element type, or -1 if the layer element type is not found.
	  */
	int GetLayerTypedIndex(int pGlobalIndex, KFbxLayerElement::ELayerElementType pType, bool pIsUV=false);
	//@}

	/** Convert Direct to Index to Direct Reference Mode.
	  * \param pLayer     The Layer to convert.
	  * \return           \c true if conversion was successful, or \c false otherwise.
	  */
	bool ConvertDirectToIndexToDirect(int pLayer);

///////////////////////////////////////////////////////////////////////////////
//
//  WARNING!
//
//	Anything beyond these lines may not be documented accurately and is 
// 	subject to change without notice.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef DOXYGEN_SHOULD_SKIP_THIS

	int  GTC(kUInt i, int j);
	void* GT (int  i,    kUInt l, int j); 
	int  AT (void* t,    kUInt l, int j);
	int  GTI(char const* n, kUInt l, int j);
	int  GMC(kUInt i, void* n = NULL);
	void* GM (int  i,    kUInt l, void* n = NULL);
	int  AM (void* m,    kUInt l, void* n = NULL);
	int  GMI(char const* n, kUInt l, void* d = NULL);

	int AddToLayerElementsList(KFbxLayerElement* pLEl);
	void RemoveFromLayerElementsList(KFbxLayerElement* pLEl);

protected:

	KFbxLayerContainer(KFbxSdkManager& pManager, char const* pName);
	virtual ~KFbxLayerContainer();

	void CopyLayers(KFbxLayerContainer const* pLayerContainer);
	KFbxLayerContainer& operator=(KFbxLayerContainer const& pLayerContainer);

	virtual void SetDocument(KFbxDocument* pDocument);
	virtual	bool ConnecNotify (KFbxConnectEvent const &pEvent);

	KArrayTemplate<KFbxLayer*> mLayerArray;
	KArrayTemplate<KFbxLayerElement*> mLayerElementsList;

	friend class KFbxScene;
	friend class KFbxGeometryConverter;
	friend class KFbxWriterFbx6;

#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS

};

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_LAYER_CONTAINER_H_


