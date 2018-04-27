#ifndef _KFbxDataTypes_h
#define _KFbxDataTypes_h

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

#ifndef MB_FBXSDK
#include <kbaselib_nsuse.h>
#endif
#include <kbaselib_forward.h>
#include <klib/kdebug.h>
#include <klib/kstring.h>

#include <kfbxplugins/kfbxtypes.h>
#include <kfbxplugins/kfbxcolor.h>
#include <kfbxmath/kfbxvector4.h>

#include <fbxcore/kfbxpropertyhandle.h>

#include <fbxfilesdk_nsbegin.h>

    class   KFbxSdkManager;
    class   KFbxTypeInfo_internal;

	/**FBX SDK data type class
	  *\nosubgrouping
	  */
    class KFBX_DLL KFbxDataType {

        public:
            static KFbxDataType Create(const char *pName,EFbxType pType);
            static KFbxDataType Create(const char *pName,KFbxDataType const &pDataType);
    /**
	  *\name Constructor and Destructor.
	  */
	//@{
			//!Constructor.
            KFbxDataType();

			//!Copy constructor.
            KFbxDataType( KFbxDataType const &pDataType );

            //!Destroy this datatype.
            void Destroy();

            /**Constructor.
			  *\param pTypeInfoHandle                Type information handle
			  */
            KFbxDataType( KFbxTypeInfoHandle const &pTypeInfoHandle );

			//!Destructor.
            ~KFbxDataType();
	//@}


        public:
			/**Assignment operator
			  *\param pDataType               Datatype whose value is assigned to this datatype.
			  *\return                        this datatype
			  */
            inline KFbxDataType& operator = (const KFbxDataType &pDataType) { mTypeInfoHandle=pDataType.mTypeInfoHandle; return *this;  }

			/**
			  * \name boolean operation
			  */
			//@{

			/**Equality operator
			  *\param pDataType                Datatype to compare to.
			  *\return                         \c true if equal,\c false otherwise.
			  */
            inline bool operator == (const KFbxDataType& pDataType) const   { return mTypeInfoHandle==pDataType.mTypeInfoHandle;        }

            /**Non-equality operator
			  *\param pDataType                Datatype to compare to.
			  *\return                         \c true if unequal,\c false otherwise.
			  */
            inline bool operator != (const KFbxDataType& pDataType) const   { return mTypeInfoHandle!=pDataType.mTypeInfoHandle;        }
            //@}
        public:

			/**Test whether this datatype is a valid datatype.
			  *\return         \c true if valid, \c false otherwise.
			  */
            inline bool     Valid() const { return mTypeInfoHandle.Valid(); }

			/** Test if this datatype is the specified datatype. 
			  * \param pDataType               Datatype to compare to.
			  * \return                        \c true if this datatype is the specified datatype, \c false otherwise. 
			  */
            inline bool     Is(const KFbxDataType& pDataType) const { return mTypeInfoHandle.Is(pDataType.mTypeInfoHandle); }

			/** Retrieve this data type.
			  * \return     this data type.
			  */
            EFbxType        GetType() const;

			/** Retrieve data type name.
              * \return     data type name.
			  */
            const char*     GetName() const;

        private:
            KFbxTypeInfoHandle      mTypeInfoHandle;
        public:
			/** Retrieve the information handle of this data type.
			  * \return       information handle of this data type.
			  */
            inline KFbxTypeInfoHandle const &GetTypeInfoHandle() const  { return mTypeInfoHandle; }

        friend class KFbxSdkManager;
    };

    // Default Basic Types
    extern KFBX_DLL  KFbxDataType   DTNone;
    extern KFBX_DLL  KFbxDataType   DTBool;
    extern KFBX_DLL  KFbxDataType   DTInteger;
    extern KFBX_DLL  KFbxDataType   DTFloat;
    extern KFBX_DLL  KFbxDataType   DTDouble;
    extern KFBX_DLL  KFbxDataType   DTDouble2;
    extern KFBX_DLL  KFbxDataType   DTDouble3;
    extern KFBX_DLL  KFbxDataType   DTDouble4;
    extern KFBX_DLL  KFbxDataType   DTDouble44;
    extern KFBX_DLL  KFbxDataType   DTEnum;
    extern KFBX_DLL  KFbxDataType   DTStringList;   // ?
    extern KFBX_DLL  KFbxDataType   DTString;
    extern KFBX_DLL  KFbxDataType   DTCharPtr;      // ?
    extern KFBX_DLL  KFbxDataType   DTTime;
    extern KFBX_DLL  KFbxDataType   DTReference;
    extern KFBX_DLL  KFbxDataType   DTCompound;
    extern KFBX_DLL  KFbxDataType   DTBlob;
	extern KFBX_DLL	 KFbxDataType	DTDistance;
    extern KFBX_DLL	 KFbxDataType	DTDateTime;

    // MB Specific datatypes
    extern KFBX_DLL  KFbxDataType   DTAction;
    extern KFBX_DLL  KFbxDataType   DTEvent;

    // Specialised reference Properties
    extern KFBX_DLL  KFbxDataType   DTReferenceObject;
    extern KFBX_DLL  KFbxDataType   DTReferenceProperty;

    // Extended sub types
    extern KFBX_DLL  KFbxDataType   DTColor3;
    extern KFBX_DLL  KFbxDataType   DTColor4;

    // Support for older datatypes
    extern KFBX_DLL  KFbxDataType   DTReal;
    extern KFBX_DLL  KFbxDataType   DTVector3D;
    extern KFBX_DLL  KFbxDataType   DTVector4D;

    // Tranforms Types
    extern KFBX_DLL  KFbxDataType   DTTranslation;
    extern KFBX_DLL  KFbxDataType   DTRotation;
    extern KFBX_DLL  KFbxDataType   DTScaling;
    extern KFBX_DLL  KFbxDataType   DTQuaternion;

    extern KFBX_DLL  KFbxDataType   DTLocalTranslation;
    extern KFBX_DLL  KFbxDataType   DTLocalRotation;
    extern KFBX_DLL  KFbxDataType   DTLocalScaling;
    extern KFBX_DLL  KFbxDataType   DTLocalQuaternion;

    extern KFBX_DLL  KFbxDataType   DTTransformMatrix;
    extern KFBX_DLL  KFbxDataType   DTTranslationMatrix;
    extern KFBX_DLL  KFbxDataType   DTRotationMatrix;
    extern KFBX_DLL  KFbxDataType   DTScalingMatrix;

    // Material-related types (also used as DataType for Texture layer elements)
    extern KFBX_DLL  KFbxDataType DTMaterialEmissive;
    extern KFBX_DLL  KFbxDataType DTMaterialEmissiveFactor;
    extern KFBX_DLL  KFbxDataType DTMaterialAmbient;
    extern KFBX_DLL  KFbxDataType DTMaterialAmbientFactor;
    extern KFBX_DLL  KFbxDataType DTMaterialDiffuse;
    extern KFBX_DLL  KFbxDataType DTMaterialDiffuseFactor;
    extern KFBX_DLL  KFbxDataType DTMaterialBump;
    extern KFBX_DLL  KFbxDataType DTMaterialNormalMap;
    extern KFBX_DLL  KFbxDataType DTMaterialTransparentColor;
    extern KFBX_DLL  KFbxDataType DTMaterialTransparencyFactor;
    extern KFBX_DLL  KFbxDataType DTMaterialSpecular;
    extern KFBX_DLL  KFbxDataType DTMaterialSpecularFactor;
    extern KFBX_DLL  KFbxDataType DTMaterialShininess;
    extern KFBX_DLL  KFbxDataType DTMaterialReflection;
    extern KFBX_DLL  KFbxDataType DTMaterialReflectionFactor;

    // LayerElement
    extern KFBX_DLL  KFbxDataType DTLayerElementUndefined;
    extern KFBX_DLL  KFbxDataType DTLayerElementNormal;
    extern KFBX_DLL  KFbxDataType DTLayerElementMaterial;
    extern KFBX_DLL  KFbxDataType DTLayerElementTexture;
    extern KFBX_DLL  KFbxDataType DTLayerElementPolygonGroup;
    extern KFBX_DLL  KFbxDataType DTLayerElementUV;
    extern KFBX_DLL  KFbxDataType DTLayerElementVertexColor;
    extern KFBX_DLL  KFbxDataType DTLayerElementSmoothing;
    extern KFBX_DLL  KFbxDataType DTLayerElementUserData;
    extern KFBX_DLL  KFbxDataType DTLayerElementVisibility;

    // File references / External file references
    extern KFBX_DLL  KFbxDataType   DTUrl;
    extern KFBX_DLL  KFbxDataType   DTXRefUrl;

    // Motion Builder Specialised Types
    extern KFBX_DLL  KFbxDataType   DTIntensity;
    extern KFBX_DLL  KFbxDataType   DTConeAngle;
    extern KFBX_DLL  KFbxDataType   DTFog;
    extern KFBX_DLL  KFbxDataType   DTShape;
    extern KFBX_DLL  KFbxDataType   DTFieldOfView;
    extern KFBX_DLL  KFbxDataType   DTFieldOfViewX;
    extern KFBX_DLL  KFbxDataType   DTFieldOfViewY;
    extern KFBX_DLL  KFbxDataType   DTOpticalCenterX;
    extern KFBX_DLL  KFbxDataType   DTOpticalCenterY;

    extern KFBX_DLL  KFbxDataType   DTRoll;
    extern KFBX_DLL  KFbxDataType   DTCameraIndex;
    extern KFBX_DLL  KFbxDataType   DTTimeWarp;
    extern KFBX_DLL  KFbxDataType   DTVisibility;

    extern KFBX_DLL  KFbxDataType   DTTranslationUV;
    extern KFBX_DLL  KFbxDataType   DTScalingUV;
    extern KFBX_DLL  KFbxDataType   DTHSB;

    extern KFBX_DLL  KFbxDataType   DTOrientation;
    extern KFBX_DLL  KFbxDataType   DTLookAt;

    extern KFBX_DLL  KFbxDataType   DTOcclusion;
    extern KFBX_DLL  KFbxDataType   DTWeight;

    extern KFBX_DLL  KFbxDataType   DTIKReachTranslation;
    extern KFBX_DLL  KFbxDataType   DTIKReachRotation;

    KFBX_DLL KFbxDataType const & GetFbxDataType(EFbxType pType);

    // Internal use
    KFBX_DLL bool               KFbxTypesInit   (KFbxSdkManager *pManager);
    KFBX_DLL void               KFbxTypesRelease(KFbxSdkManager *pManager);

    KFBX_DLL char*              FbxDataTypeToFormatPropertyType( KFbxDataType const &pDataType );
    KFBX_DLL KFbxDataType const &FormatPropertyTypeToFbxDataType( const char *pDataTypeName );



#include <fbxfilesdk_nsend.h>

#endif // _KFbxTypes_h


