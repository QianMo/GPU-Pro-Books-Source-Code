/*!  \file kfbxproperty.h
 */
#ifndef _KFbxProperty_h
#define _KFbxProperty_h

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

// FBX includes
#include <kfcurve/kfcurvenode.h>
#include <fbxcore/kfbxevaluationinfo.h>
#include <kfbxplugins/kfbxdatatypes.h>
#include <fbxcore/kfbxpropertydef.h>

// FBX namespace
#include <fbxfilesdk_nsbegin.h>

// Forward declarations
class KFbxObject;
class KFbxColor;
class KFbxCriteria;

    /** \brief Class to hold user properties.
      * \nosubgrouping
      */
    class KFBX_DLL KFbxProperty : private FbxPropertyFlags
    {
        /**
          * \name Constructor and Destructor.
          */
        //@{
        public:
            using FbxPropertyFlags::eFbxPropertyFlags;
            using FbxPropertyFlags::eNO_FLAG;
            using FbxPropertyFlags::eANIMATABLE;
            using FbxPropertyFlags::eUSER;
            using FbxPropertyFlags::eTEMPORARY;
            using FbxPropertyFlags::ePUBLISHED;
            using FbxPropertyFlags::ePSTATIC;

            using FbxPropertyFlags::eNOT_SAVABLE;
            using FbxPropertyFlags::eHIDDEN;

            using FbxPropertyFlags::eUI_DISABLED;
            using FbxPropertyFlags::eUI_GROUP;
			using FbxPropertyFlags::eUI_BOOLGROUP;
			using FbxPropertyFlags::eUI_EXPANDED;
			using FbxPropertyFlags::eUI_NOCAPTION;
			using FbxPropertyFlags::eUI_PANEL;

            typedef FbxPropertyFlags::eFbxPropertyFlags EFlags;

            /** Create a property
              * \param pCompoundProperty
              * \param pName
              * \param pDataType
              * \param pLabel
              * \param pCheckForDuplicate
              * \param pWasFound
              */
            static KFbxProperty Create(KFbxProperty const &pCompoundProperty, char const* pName, KFbxDataType const &pDataType=KFbxDataType(), char const* pLabel="",bool pCheckForDuplicate=true, bool* pWasFound=NULL);

            /** Create a property
              * \param pParentProperty
              * \param pName
              * \param pDataType
              * \param pValue
              * \param pFlags
              * \param pCheckForDuplicate
              * \param pForceSet
              */
            template<typename T> inline static KFbxProperty Create(KFbxProperty const &pParentProperty, char const* pName,KFbxDataType const &pDataType,T const &pValue,eFbxPropertyFlags pFlags=eNO_FLAG,bool pCheckForDuplicate=true,bool pForceSet=true)
            {
                if( !pCheckForDuplicate )
                {
                    KFbxProperty lProperty = Create(pParentProperty, pName, pDataType, "", pCheckForDuplicate);
                    lProperty.ModifyFlag(pFlags, true); // modify the flags before we set the value
                    lProperty.Set(pValue);
                    return lProperty;
                }

                // First check for a duplicate
                KFbxProperty lProperty = pParentProperty.Find(pName);

                // only set if we are forcing the set, or we are actually creating the property
                // (as opposed to returning an existing one)
                //bool lSetValue = pForceSet ? true : !lProperty.IsValid();

                if( !lProperty.IsValid() )
                    lProperty = Create(pParentProperty, pName, pDataType, "", false); // don't check because we already did

                lProperty.ModifyFlag(pFlags, true); // modify the flags before we set the value
                //if( lSetValue )
                //  lProperty.Set(pValue);
                lProperty.Set( &pValue,FbxTypeOf(pValue),!pForceSet);
                return lProperty;
            }

            /** Create a dynamic property
              * \param pObject
              * \param pName
              * \param pDataType
              * \param pLabel
              * \param pCheckForDuplicate
              * \param pWasFound
              */
            static KFbxProperty Create(KFbxObject* pObject, char const* pName, KFbxDataType const &pDataType=KFbxDataType(), char const* pLabel="",bool pCheckForDuplicate=true,bool* pWasFound=NULL);

            /** Create a dynamic property from an other property
              * \param pObject
              * \param pFromProperty
              * \param pCheckForDuplicate
              */
            static KFbxProperty Create(KFbxObject* pObject, KFbxProperty& pFromProperty, bool pCheckForDuplicate=true);

         /** Create a dynamic property from an other property
           * \param pCompoundProperty
           * \param pFromProperty
           * \param pCheckForDuplicate
           */
         static KFbxProperty Create(KFbxProperty const& pCompoundProperty, KFbxProperty& pFromProperty, bool pCheckForDuplicate = true);

            /** Destroy a dynamic property
              */
            void Destroy(bool pRecursive = true, bool pDependents = false);

            /** Static Property Constructors
              */
            KFbxProperty();

            /** Copy constructor for properties
              */
            KFbxProperty(KFbxProperty const &pProperty);

            /** Copy constructor for properties
              */
            KFbxProperty(KFbxPropertyHandle const &pProperty);

            /** Static Property destructor
              */
            ~KFbxProperty();

        public:
        //@}

        /**
          * \name Property Identification.
          */
        //@{
        public:
            /** Get the property data type definition.
              * \return The properties KFbxDataType
              */
            KFbxDataType        GetPropertyDataType() const;

            /** Get the property internal name.
              * \return Property internal name string.
              */
            KString         GetName() const;

            /** Get the property internal name.
              * \return Property internal name string.
              */
            KString             GetHierarchicalName() const;

            /** Get the property label.
              * \param pReturnNameIfEmpty If \c true, allow this method to return the internal name.
              * \return The property label if set, or the property internal name if the pReturnNameIfEmpty
              * flag is set to \c true and the label has not been defined.
              * \remarks Some applications may choose to ignore the label field and work uniquely with the
              * internal name. Therefore, it should not be taken for granted that a label exists. Also, remember
              * that the label does not get saved in the FBX file. It only exist while the property object is
              * in memory.
              */
            KString         GetLabel(bool pReturnNameIfEmpty=true);


            /** Set a label to the property.
              * \param pLabel Label string.
              */
            void                SetLabel(KString pLabel);

            /** Get the object that contains the property.
              * \return the property object owner or null if the property is an orphan.
              */
            KFbxObject*         GetFbxObject() const;

        //@}

        /**
          * \name User data
          */
        //@{
            void                SetUserTag(int pTag);
            int                 GetUserTag();
            void  SetUserDataPtr(void* pUserData);
            void* GetUserDataPtr();
        //@}

        /**
          * \name Property Flags.
          */
        //@{
            /** Change the attributes of the property.
              * \param pFlag Property attribute identifier.
              * \param pValue New state.
              */
            void ModifyFlag(eFbxPropertyFlags pFlag, bool pValue);

            /** Get the property attribute state.
              * \param pFlag Property attribute identifier.
              * \return The currently set property attribute state.
              */
            bool GetFlag(eFbxPropertyFlags pFlag);

            /** Gets the inheritance type for the given flag. Similar to GetValueInheritType().
              * \param pFlag The flag to query
              * \return The inheritance type of the given flag
              */
            KFbxInheritType GetFlagInheritType( eFbxPropertyFlags pFlag ) const;

            /** Sets the inheritance type for the given flag. Similar to SetValueInheritType().
              * \param pFlag The flag to set
              * \param pType The inheritance type to set
              * \return true on success, false otherwise.
              */
            bool SetFlagInheritType( eFbxPropertyFlags pFlag, KFbxInheritType pType );

            /** Checks if the property's flag has been modified from its default value.
              * \param pFlag The flag to query
              * \return true if the value of this property has changed, false otherwise
              */
            bool ModifiedFlag( eFbxPropertyFlags pFlag ) const;
        //@}

        /**
          * \name Assignment and comparison operators
          */
        //@{
            KFbxProperty &      operator=  (KFbxProperty const &pKProperty);
            bool                operator== (KFbxProperty const &pKProperty) const;
            bool                operator!= (KFbxProperty const &pKProperty) const;
            inline bool         operator== (int pValue) const { return pValue==0 ? !IsValid() : IsValid(); }
            inline bool         operator!= (int pValue) const { return pValue!=0 ? !IsValid() : IsValid(); }
            bool CompareValue(KFbxProperty const& pProp) const;
        //@}

        /** Copy value of a property.
          * \param pProp Property to get value from.
          * \return true if value has been copied, false if not.
          */
        bool CopyValue(KFbxProperty const& pProp);

        /**
          * \name Value management.
          */
        //@{
        public:
            bool            IsValid() const;

            /** set value function
              * \param pValue Pointer to the new value
              * \param pValueType The data type of the new value
              * \param pCheckForValueEquality if true, the value is not set if it is equal to the default value.
              * \return true if it was succesfull and type were compatible.
              */
            bool Set(void const *pValue,EFbxType pValueType, bool pCheckForValueEquality);
            inline bool Set(void const *pValue,EFbxType pValueType) { return Set( pValue, pValueType, true ); }

            /** get value function
              * \return true if it was succesfull and type were compatible.
              */
            bool Get(void *pValue,EFbxType pValueType) const;

            /** get and evaluate pulls on a value
              * \return true if it was succesfull and type were compatible.
              */
            bool Get(void *pValue,EFbxType pValueType,KFbxEvaluationInfo const *pEvaluateInfo);

            // usefull set and get functions
            template <class T> inline bool  Set( T const &pValue )  { return Set( &pValue,FbxTypeOf(pValue), true ); }
            template <class T> inline T     Get( T const *pFBX_TYPE) const { T lValue; Get( &lValue,FbxTypeOf(lValue) ); return lValue; }
            template <class T> inline T     Get( T const *pFBX_TYPE,KFbxEvaluationInfo const *pEvaluateInfo) { T lValue; Get( &lValue,FbxTypeOf(lValue),pEvaluateInfo ); return lValue; }
            template <class T> inline T     Get( KFbxEvaluationInfo const *pEvaluateInfo) { T lValue; Get( &lValue,FbxTypeOf(lValue),pEvaluateInfo ); return lValue; }
            /** get and evaluate pulls on a value
              * \return true if it was succesfull and type were compatible.
              */
            bool Get(void *pValue,EFbxType pValueType,KFbxEvaluationInfo *pEvaluateInfo) const;

            /** Query the inheritance type of the property.
              * Use this method to determine if this property's value is overriden from the default
              * value, or from the referenced object, if this object is a clone.
              * \return The inheritance type of the property.
              */
            KFbxInheritType GetValueInheritType() const;

            /** Set the inheritance type of the property.
              * Use the method to explicitly override the default value of the property,
              * or the referenced object's property value, if this object is a clone.
              *
              * It can also be used to explicitly inherit the default value of the property,
              * or the referenced object's property value, if this object is a clone.
              *
              * \param pType The new inheritance type.
              * \return true on success, false otherwise.
              */
            bool SetValueInheritType( KFbxInheritType pType );

            /** Checks if the property's value has been modified from its default value.
              * \return true if the value of this property has changed, false otherwise
              */
            bool Modified() const;

        //@}

        /**
          * \name Property Limits.
          * Property limits are provided for convenience if some applications desire to
          * bound the range of possible values for a given type property. Note that these
          * limits are meaningless for the boolean type. It is the responsibility of the
          * calling application to implement the necessary instructions to limit the property.
          */
        //@{
        public:
            /** Set the minimum limit value of the property.
              * \param pMin Minimum value allowed.
              */
            void                SetMinLimit(double pMin);

         /** Returns whether a limit exists; calling GetMinLimit() when this returns
           * false will return in undefined behavior.
           * \return Whether or not a minimum limit has been set.
           */
         bool           HasMinLimit() const;

            /** Get the minimum limit value of the property.
              * \return Currently set minimum limit value.
              */
            double          GetMinLimit() const;

         /** Returns whether a limit exists; calling GetMinLimit() when this returns
           * false will return in undefined behavior.
           * \return Whether or not a minimum limit has been set.
           */
         bool           HasMaxLimit() const;

            /** Set the maximum limit value of the property.
              * \param pMax Maximum value allowed.
              */
            void                SetMaxLimit(double pMax);
            /** Get the maximum limit value of the property.
              * \return Currently set maximum limit value.
              */
            double          GetMaxLimit() const;

            /** Set the minimum and maximum limit value of the property.
              * \param pMin Minimum value allowed.
              * \param pMax Maximum value allowed.
              */
            void                SetLimits(double pMin, double pMax);
        //@}

        /**
          * \name Enum and property list
          */
        //@{
        public:
            /** Add a string value at the end of the list.
              * \param pStringValue Value of the string to be added.
              * \return The index in the list where the string was added.
              * \remarks This function is only valid when the property type is eENUM.
              * Empty strings are not allowed.
              */
            int                 AddEnumValue(char const *pStringValue);

            /** Insert a string value at the specified index.
              * \param pIndex Zero bound index.
              * \param pStringValue Value of the string for the specified index.
              * \remarks This function is only valid when the property type is eENUM.
              * pIndex must be in the range [0, ListValueGetCount()].
              * Empty strings are not allowed.
              */
            void                InsertEnumValue(int pIndex, char const *pStringValue);

            /** Get the number of elements in the list.
              * \return The number of elements in the list.
              * \remarks This function will return -1 if the property type is not eENUM.
              */
            int                 GetEnumCount();

            /** Set a string value for the specified index.
              * \param pIndex Zero bound index.
              * \param pStringValue Value of the string for the specified index.
              * \remarks This function is only valid when the property type is eENUM.
              * The function will assign the specified string to the specified index.
              * A string value must exists at the specified index in order to be changed.
              * Empty strings are not allowed.
              */
            void                SetEnumValue(int pIndex, char const *pStringValue);

            /** Remove the string value at the specified index.
              * \param pIndex of the string value to be removed.
              */
            void                RemoveEnumValue(int pIndex);

            /** Get a string value for the specified index
              * \param pIndex Zero bound index.
              * \remarks This function is only valid when the property type is eENUM.
              */
            char *              GetEnumValue(int pIndex);
        //@}

        /**
          * \name Hierarchical properties
          */
        //@{
            inline bool                 IsRoot() const                                          { return mPropertyHandle.IsRoot(); }
            inline bool                 IsChildOf(KFbxProperty  const & pParent) const          { return mPropertyHandle.IsChildOf(pParent.mPropertyHandle); }
            inline bool                 IsDescendentOf(KFbxProperty const & pAncestor) const    { return mPropertyHandle.IsDescendentOf(pAncestor.mPropertyHandle); }
            inline KFbxProperty         GetParent() const                                       { return KFbxProperty(mPropertyHandle.GetParent());  }
            bool                        SetParent( const KFbxProperty& pOther );
            inline KFbxProperty         GetChild() const                                        { return KFbxProperty(mPropertyHandle.GetChild());   }
            inline KFbxProperty         GetSibling() const                                      { return KFbxProperty(mPropertyHandle.GetSibling()); }

            /** Get the first property that is a descendent to this property
              * \return A valid KFbxProperty if the property was found, else
              *         an invalid KFbxProperty. See KFbxProperty::IsValid()
              */
            inline KFbxProperty         GetFirstDescendent() const                              { return KFbxProperty(mPropertyHandle.GetFirstDescendent());   }
            /** Get the next property following pProperty that is a descendent to this property
              * \param pProperty The last found descendent.
              * \return A valid KFbxProperty if the property was found, else
              *         an invalid KFbxProperty. See KFbxProperty::IsValid()
              */
            inline KFbxProperty         GetNextDescendent(KFbxProperty const &pProperty) const  { return KFbxProperty(mPropertyHandle.GetNextDescendent(pProperty.mPropertyHandle)); }

            /** Find a property using its name and its data type.
			  * \param pCaseSensitive
              * \param pName The name of the property as a \c NULL terminated string.
              * \return A valid KFbxProperty if the property was found, else
              *         an invalid KFbxProperty. See KFbxProperty::IsValid()
              */
            inline KFbxProperty         Find (char const *pName,bool pCaseSensitive = true) const { return KFbxProperty(mPropertyHandle.Find(pName,pCaseSensitive));  }
            inline KFbxProperty         Find (char const *pName,KFbxDataType const &pDataType, bool pCaseSensitive = true) const { return KFbxProperty(mPropertyHandle.Find(pName,pDataType.GetTypeInfoHandle(),pCaseSensitive));  }
            /** Fullname find
			  * \param pCaseSensitive
              * \param pName The name of the property as a \c NULL terminated string.
              * \return A valid KFbxProperty if the property was found, else
              *         an invalid KFbxProperty. See KFbxProperty::IsValid()
              */
            inline KFbxProperty         FindHierarchical (char const *pName,bool pCaseSensitive = true) const { return KFbxProperty(mPropertyHandle.Find(pName,sHierarchicalSeparator,pCaseSensitive));  }
            inline KFbxProperty         FindHierarchical (char const *pName,KFbxDataType const &pDataType, bool pCaseSensitive = true) const { return KFbxProperty(mPropertyHandle.Find(pName,sHierarchicalSeparator,pDataType.GetTypeInfoHandle(),pCaseSensitive));  }

        //@}

        /**
          * \name Optimizations
          */
        //@{
            inline void     BeginCreateOrFindProperty() { mPropertyHandle.BeginCreateOrFindProperty();  }
            inline void     EndCreateOrFindProperty()   { mPropertyHandle.EndCreateOrFindProperty();    }

         struct KFbxPropertyNameCache
         {
            KFbxPropertyNameCache(const KFbxProperty& prop) :
               mProp(const_cast<KFbxProperty&>(prop))
            {
               mProp.BeginCreateOrFindProperty();
            }

            ~KFbxPropertyNameCache()
            {
               mProp.EndCreateOrFindProperty();
            }

         private:
            KFbxProperty & mProp;

			KFbxPropertyNameCache& operator=(const KFbxPropertyNameCache &other) { mProp = other.mProp; mProp.BeginCreateOrFindProperty(); return *this; }
         };
        //@}

        /**
          * \name Array Management
          */
        //@{
            bool            SetArraySize( int pSize, bool pVariableArray );
            int             GetArraySize() const;
            KFbxProperty    GetArrayItem(int pIndex) const;
            inline KFbxProperty operator[](int pIndex) const { return GetArrayItem(pIndex); }
        //@}

        /**
          * \name FCurve Management
          */
        //@{
            /** Create a KFCurveNode on a take
              * \param pTakeName Name of the take to create the KFCurveNode on
              */
            KFCurveNode* CreateKFCurveNode(const char* pTakeName=NULL);

            /** Get the KFCurveNode from a take
              * \param pTakeName Name of the take to get the KFCurveNode from
              * \param pCreateAsNeeded Create the KFCurveNode if not found.
              * \return Pointer to the KFCurveNode of the proprety on the given take.
              */
            KFCurveNode* GetKFCurveNode(bool pCreateAsNeeded=false, const char* pTakeName=NULL);

            /** Tries to get the KFCurve of the specified channel from the current take.
              * \param pChannel Name of the fcurve channel we are looking for.
              * \return Pointer to the FCurve if found, NULL in any other case.
              * \remark This method will fail if the KFCurveNode does not exist.
              * \remark If the pChannel is left NULL, this method retrieve the FCurve directly from the KFCurveNode
              * otherwise it will look recursively to find it.
             */
            KFCurve* GetKFCurve(const char* pChannel = NULL);

            /** Tries to get the KFCurve of the specified channel from the given take.
              * \param pChannel Name of the fcurve channel we are looking for.
              * \param pTakeName Name of the take to get the KFCurve from.
              * \return Pointer to the FCurve if found, NULL in any other case.
              * \remark This method will fail if the KFCurveNode does not exist.
              * \remark If pTakeName is NULL, this function will look in the current take.
              * \remark If the pChannel is left NULL, this method retrieve the FCurve directly from the KFCurveNode
              * otherwise it will look recursively to find it.
             */
            KFCurve* GetKFCurve(const char* pChannel, const char* pTakeName);
        //@}

        /**
          * \name Evaluation management
          */
        //@{
            bool    Evaluate(KFbxEvaluationInfo const *pEvaluationInfo);
        //@}

        /**
          * \name General Object Connection and Relationship Management
          */
        //@{
        public:
            // SrcObjects
            bool ConnectSrcObject       (KFbxObject* pObject,kFbxConnectionType pType=eFbxConnectionNone);
            bool IsConnectedSrcObject   (const KFbxObject* pObject) const;
            bool DisconnectSrcObject    (KFbxObject* pObject);

            bool DisconnectAllSrcObject();
            bool DisconnectAllSrcObject(KFbxCriteria const &pCriteria);
            bool DisconnectAllSrcObject(const kFbxClassId& pClassId);
            bool DisconnectAllSrcObject(const kFbxClassId& pClassId,KFbxCriteria const &pCriteria);

            int GetSrcObjectCount()const;
            int GetSrcObjectCount(KFbxCriteria const &pCriteria)const;
            int GetSrcObjectCount(const kFbxClassId& pClassId)const;
            int GetSrcObjectCount(const kFbxClassId& pClassId,KFbxCriteria const &pCriteria)const;

            KFbxObject* GetSrcObject(int pIndex=0) const;
            KFbxObject* GetSrcObject(KFbxCriteria const &pCriteria,int pIndex=0) const;
            KFbxObject* GetSrcObject(const kFbxClassId& pClassId,int pIndex=0) const;
            KFbxObject* GetSrcObject(const kFbxClassId& pClassId,KFbxCriteria const &pCriteria,int pIndex=0) const;

            KFbxObject* FindSrcObject(const char *pName,int pStartIndex=0) const;
            KFbxObject* FindSrcObject(KFbxCriteria const &pCriteria,const char *pName,int pStartIndex=0) const;
            KFbxObject* FindSrcObject(const kFbxClassId& pClassId,const char *pName,int pStartIndex=0) const;
            KFbxObject* FindSrcObject(const kFbxClassId& pClassId,KFbxCriteria const &pCriteria,const char *pName,int pStartIndex=0) const;

            template < class T > inline bool DisconnectAllSrcObject (T const *pFBX_TYPE){ return DisconnectAllSrcObject(T::ClassId);}
            template < class T > inline bool DisconnectAllSrcObject (T const *pFBX_TYPE,KFbxCriteria const &pCriteria)  { return DisconnectAllSrcObject(T::ClassId,pCriteria);  }
            template < class T > inline int  GetSrcObjectCount(T const *pFBX_TYPE) const{ return GetSrcObjectCount(T::ClassId); }
            template < class T > inline int  GetSrcObjectCount(T const *pFBX_TYPE,KFbxCriteria const &pCriteria) const { return GetSrcObjectCount(T::ClassId,pCriteria); }
            template < class T > inline T*   GetSrcObject(T const *pFBX_TYPE,int pIndex=0) const { return (T*)GetSrcObject(T::ClassId,pIndex); }
            template < class T > inline T*   GetSrcObject(T const *pFBX_TYPE,KFbxCriteria const &pCriteria,int pIndex=0) const { return (T*)GetSrcObject(T::ClassId,pCriteria,pIndex); }
            template < class T > inline T*   FindSrcObject(T const *pFBX_TYPE,const char *pName,int pStartIndex=0) const { return (T*)FindSrcObject(T::ClassId,pName,pStartIndex); }
            template < class T > inline T*   FindSrcObject(T const *pFBX_TYPE,KFbxCriteria const &pCriteria,const char *pName,int pStartIndex=0) const { return (T*)FindSrcObject(T::ClassId,pCriteria,pName,pStartIndex); }

            // DstObjects
            bool ConnectDstObject       (KFbxObject* pObject,kFbxConnectionType pType=eFbxConnectionNone);
            bool IsConnectedDstObject   (const KFbxObject* pObject) const;
            bool DisconnectDstObject    (KFbxObject* pObject);

            bool DisconnectAllDstObject();
            bool DisconnectAllDstObject(KFbxCriteria const &pCriteria);
            bool DisconnectAllDstObject(const kFbxClassId& pClassId);
            bool DisconnectAllDstObject(const kFbxClassId& pClassId,KFbxCriteria const &pCriteria);

            int GetDstObjectCount() const;
            int GetDstObjectCount(KFbxCriteria const &pCriteria) const;
            int GetDstObjectCount(const kFbxClassId& pClassId) const;
            int GetDstObjectCount(const kFbxClassId& pClassId,KFbxCriteria const &pCriteria) const;

            KFbxObject* GetDstObject(int pIndex=0) const;
            KFbxObject* GetDstObject(KFbxCriteria const &pCriteria,int pIndex=0) const;
            KFbxObject* GetDstObject(const kFbxClassId& pClassId,int pIndex=0) const;
            KFbxObject* GetDstObject(const kFbxClassId& pClassId,KFbxCriteria const &pCriteria,int pIndex=0)const;

            KFbxObject* FindDstObject(const char *pName,int pStartIndex=0) const;
            KFbxObject* FindDstObject(KFbxCriteria const &pCriteria,const char *pName,int pStartIndex=0) const;
            KFbxObject* FindDstObject(const kFbxClassId& pClassId,const char *pName,int pStartIndex=0) const;
            KFbxObject* FindDstObject(const kFbxClassId& pClassId,KFbxCriteria const &pCriteria,const char *pName,int pStartIndex=0) const;

            template < class T > inline bool DisconnectAllDstObject (T const *pFBX_TYPE){ return DisconnectAllDstObject(T::ClassId);    }
            template < class T > inline bool DisconnectAllDstObject (T const *pFBX_TYPE,KFbxCriteria const &pCriteria)  { return DisconnectAllDstObject(T::ClassId,pCriteria);  }
            template < class T > inline int  GetDstObjectCount(T const *pFBX_TYPE) const { return GetDstObjectCount(T::ClassId); }
            template < class T > inline int  GetDstObjectCount(T const *pFBX_TYPE,KFbxCriteria const &pCriteria) const { return GetDstObjectCount(T::ClassId,pCriteria); }
            template < class T > inline T*   GetDstObject(T const *pFBX_TYPE,int pIndex=0) const { return (T*)GetDstObject(T::ClassId,pIndex); }
            template < class T > inline T*   GetDstObject(T const *pFBX_TYPE,KFbxCriteria const &pCriteria,int pIndex=0) const { return (T*)GetDstObject(T::ClassId,pCriteria,pIndex); }
            template < class T > inline T*   FindDstObject(T const *pFBX_TYPE,const char *pName,int pStartIndex=0) const { return (T*)FindDstObject(T::ClassId,pName,pStartIndex); }
            template < class T > inline T*   FindDstObject(T const *pFBX_TYPE,KFbxCriteria const &pCriteria,const char *pName,int pStartIndex=0) const { return (T*)FindDstObject(T::ClassId,pCriteria,pName,pStartIndex); }
        //@}

        /**
          * \name General Property Connection and Relationship Management
          */
        //@{
        public:
            // Properties
            bool            ConnectSrcProperty      (KFbxProperty const & pProperty);
            bool            IsConnectedSrcProperty  (KFbxProperty const & pProperty);
            bool            DisconnectSrcProperty   (KFbxProperty const & pProperty);
            int             GetSrcPropertyCount     () const;
            KFbxProperty    GetSrcProperty          (int pIndex=0) const;
            KFbxProperty    FindSrcProperty         (const char *pName,int pStartIndex=0) const;

            bool            ConnectDstProperty      (KFbxProperty const & pProperty);
            bool            IsConnectedDstProperty  (KFbxProperty const & pProperty);
            bool            DisconnectDstProperty   (KFbxProperty const & pProperty);
            int             GetDstPropertyCount     () const;
            KFbxProperty    GetDstProperty          (int pIndex=0) const;
            KFbxProperty    FindDstProperty         (const char *pName,int pStartIndex=0) const;

            void            ClearConnectCache();

        //@}

        static const char* sHierarchicalSeparator;

        // Deprecated function calls
        typedef enum {   eUNIDENTIFIED,eBOOL,eREAL,eCOLOR,eINTEGER,eVECTOR,eLIST, eMATRIX} EUserPropertyType;

        K_DEPRECATED static const char* GetPropertyTypeName(EUserPropertyType pType);
        K_DEPRECATED const char *       GetPropertyTypeName();
        K_DEPRECATED EUserPropertyType  GetPropertyType();

        K_DEPRECATED void SetDefaultValue(bool pValue);
        K_DEPRECATED void SetDefaultValue(double pValue);
        K_DEPRECATED void SetDefaultValue(KFbxColor& pValue);
        K_DEPRECATED void SetDefaultValue(int pValue);
        K_DEPRECATED void SetDefaultValue(double pValue1, double pValue2, double pValue3);
        K_DEPRECATED void GetDefaultValue(bool& pValue);
        K_DEPRECATED void GetDefaultValue(double& pValue);
        K_DEPRECATED void GetDefaultValue(KFbxColor& pValue);
        K_DEPRECATED void GetDefaultValue(int& pValue);
        K_DEPRECATED void GetDefaultValue(double& pValue1, double& pValue2, double& pValue3);

        ///////////////////////////////////////////////////////////////////////////////
        //  WARNING!
        //  Anything beyond these lines may not be documented accurately and is
        //  subject to change without notice.
        ///////////////////////////////////////////////////////////////////////////////
        #ifndef DOXYGEN_SHOULD_SKIP_THIS

        protected:
            //! Constructor / Destructor
            KFbxProperty(KFbxObject* pObject, char const* pName, KFbxDataType const &pDataType=KFbxDataType(), char const* pLabel="");
            KFbxProperty(KFbxProperty const & pParent, char const* pName, KFbxDataType const &pDataType, char const* pLabel);

        // General Property Connection and Relationship Management
        private:
            bool            ConnectSrc      (KFbxProperty const &pProperty,kFbxConnectionType pType=eFbxConnectionNone);
            bool            DisconnectSrc   (KFbxProperty const &pProperty);
            bool            DisconnectAllSrc();
            bool            DisconnectAllSrc(KFbxCriteria const &pCriteria);
            bool            IsConnectedSrc  (KFbxProperty const &pProperty) const;
            int             GetSrcCount     () const;
            int             GetSrcCount     (KFbxCriteria const &pCriteria) const;
            KFbxProperty    GetSrc          (int pIndex=0) const;
            KFbxProperty    GetSrc          (KFbxCriteria const &pCriteria,int pIndex=0) const;
            KFbxProperty    FindSrc         (KFbxCriteria const &pCriteria,const char *pName,int pStartIndex=0) const;

            bool            ConnectDst      (KFbxProperty const &pProperty,kFbxConnectionType pType=eFbxConnectionNone);
            bool            DisconnectDst   (KFbxProperty const &pProperty);
            bool            DisconnectAllDst();
            bool            DisconnectAllDst(KFbxCriteria const &pCriteria);
            bool            IsConnectedDst  (KFbxProperty const &pProperty) const;
            int             GetDstCount     () const;
            int             GetDstCount     (KFbxCriteria const &pCriteria) const;
            KFbxProperty    GetDst          (int pIndex=0) const;
            KFbxProperty    GetDst          (KFbxCriteria const &pCriteria,int pIndex=0) const;
            KFbxProperty    FindDst         (KFbxCriteria const &pCriteria,const char *pName,int pStartIndex=0) const;
        private:
            //! Internal management
            mutable KFbxPropertyHandle  mPropertyHandle;

            friend class KFbxObject;
        #endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS
    };

    typedef KFbxProperty* HKFbxProperty;

    template <class T> inline bool  KFbxSet( KFbxProperty &pProperty,T const &pValue, bool pCheckValueEquality = true ) { return pProperty.Set( &pValue,FbxTypeOf(pValue),pCheckValueEquality ); }
    template <class T> inline bool  KFbxGet( KFbxProperty const& pProperty,T &pValue ) { return pProperty.Get( &pValue,FbxTypeOf(pValue) ); }
    template <class T> inline bool  KFbxGet( KFbxProperty& pProperty, T &pValue, KFbxEvaluationInfo const* pInfo)  { return pProperty.Get( &pValue, FbxTypeOf(pValue), pInfo ); }
    template <class T> inline T     KFbxGet( KFbxProperty const& pProperty)            { T pValue; pProperty.Get( &pValue,FbxTypeOf(pValue) ); return pValue; }


    template <class T> class KFbxTypedProperty : public KFbxProperty {
        public:
            inline KFbxTypedProperty() : KFbxProperty()                                         {}
            inline KFbxTypedProperty(KFbxProperty const &pProperty) : KFbxProperty(pProperty)   {}
            inline ~KFbxTypedProperty() {}

            inline KFbxProperty const & StaticInit(KFbxObject* pObject, char const* pName,eFbxPropertyFlags pFlags=eNO_FLAG)
            {
                *this = Create(pObject, pName, GetFbxDataType(FbxTypeOf(*((T *)0))), "");
                ModifyFlag(pFlags, true);
                return *this;
            }

            inline KFbxProperty const & StaticInit(KFbxObject* pObject, char const* pName,T const &pValue,bool pForceSet=true,eFbxPropertyFlags pFlags=eNO_FLAG)
            {
                bool lWasFound = false;
                *this = Create(pObject, pName, GetFbxDataType(FbxTypeOf(*((T *)0))), "",true,&lWasFound);

                if( pForceSet || !lWasFound )
                {
                    ModifyFlag(pFlags, true); // modify the flags before we set the value
                    Set(pValue,false);
                }

                return *this;
            }
            inline KFbxProperty const & StaticInit(KFbxObject* pObject, char const* pName, KFbxDataType const &pDataType,eFbxPropertyFlags pFlags=eNO_FLAG)
            {
                *this = Create(pObject, pName, pDataType, "");
                ModifyFlag(pFlags, true);
                return *this;
            }
            inline KFbxProperty const & StaticInit(KFbxObject* pObject, char const* pName, KFbxDataType const &pDataType,T const &pValue, bool pForceSet=true, eFbxPropertyFlags pFlags=eNO_FLAG)
            {
                bool lWasFound = false;
                *this = Create(pObject, pName, pDataType, "",true,&lWasFound);

                if( pForceSet || !lWasFound )
                {
                    ModifyFlag(pFlags, true); // modify the flags before we set the value
                    // since we will trigger callbacks in there!
                    Set(pValue,false);
                }

                return *this;
            }

            inline KFbxProperty const & StaticInit(KFbxProperty pCompound, char const* pName, KFbxDataType const &pDataType,T const &pValue, bool pForceSet=true, eFbxPropertyFlags pFlags=eNO_FLAG)
            {
                bool lWasFound = false;
                *this = Create(pCompound, pName, pDataType, "",true,&lWasFound);

                if( pForceSet || !lWasFound )
                {
                    ModifyFlag(pFlags, true); // modify the flags before we set the value
                    // since we will trigger callbacks in there!
                    Set(pValue,false);
                }

                return *this;
            }
        public:
            KFbxTypedProperty &operator =(T const &pValue)      { KFbxSet(*this,pValue); return *this; }
            bool     Set(T const &pValue, bool pCheckValueEquality )    { return KFbxSet(*this,pValue,pCheckValueEquality); }
            bool     Set(T const &pValue )  { return KFbxSet(*this,pValue,true); }
            T        Get() const            { T lValue; KFbxGet(*this,lValue); return lValue; }
            T        Get( KFbxEvaluationInfo const* pInfo ) { T lValue; KFbxGet( *this, lValue, pInfo); return lValue; }

        friend class KFbxObject;
    };

    // For use with deprecated type functions
    KFBX_DLL KFbxDataType                       EUserPropertyTypeToDataType(KFbxProperty::EUserPropertyType);
    KFBX_DLL KFbxProperty::EUserPropertyType        DataTypeToEUserPropertyType(const KFbxDataType &pDataType);


    template <> class KFbxTypedProperty<fbxReference*> : public KFbxProperty
    {
    public:
        inline KFbxTypedProperty() : KFbxProperty()
        {}
        inline KFbxTypedProperty(KFbxProperty const &pProperty)
            : KFbxProperty(pProperty)
        {}
        inline ~KFbxTypedProperty()
        {}

        inline KFbxProperty const & StaticInit(KFbxObject* pObject,char const* pName,eFbxPropertyFlags pFlags=eNO_FLAG)
        {
            *this = KFbxProperty::Create(pObject, pName, GetFbxDataType(FbxTypeOf(*((fbxReference* *)0))), "");
            ModifyFlag(pFlags, true);
            return *this;
        }

        inline KFbxProperty const & StaticInit(KFbxObject* pObject,
                                               char const* pName,
                                               fbxReference* const &pValue,
                                               bool pForceSet=true,
                                               eFbxPropertyFlags pFlags=eNO_FLAG
                                               )
        {
            bool lWasFound = false;
            *this = KFbxProperty::Create(pObject, pName, GetFbxDataType(FbxTypeOf(*((fbxReference* *)0))), "",true, &lWasFound);
            if( pForceSet || !lWasFound )
            {
                ModifyFlag(pFlags, true);
                Set(pValue,false);
            }

            return *this;
        }

        inline KFbxProperty const & StaticInit(KFbxObject* pObject,
                                               char const* pName,
                                               KFbxDataType const &pDataType,
                                               eFbxPropertyFlags pFlags=eNO_FLAG)
        {
            *this = KFbxProperty::Create(pObject, pName, pDataType, "");
//          KFbxProperty::StaticInit(pObject, pName, pDataType, "");
            ModifyFlag(pFlags, true);
            return *this;
        }

        inline KFbxProperty const & StaticInit(KFbxObject* pObject,
                                               char const* pName,
                                               KFbxDataType const &pDataType,
                                               fbxReference* const &pValue,
                                               bool pForceSet=true,
                                               eFbxPropertyFlags pFlags=eNO_FLAG
                                               )
        {
            bool lWasFound = false;
            *this = KFbxProperty::Create(pObject, pName, pDataType, "",true,&lWasFound);

            if( pForceSet || !lWasFound )
            {
                ModifyFlag(pFlags, true);
                Set(pValue,false);
            }

            return *this;
        }

    public:
        KFbxTypedProperty &operator =(fbxReference* const &pValue)
        {
            KFbxSet(*this,pValue);
            return *this;
        }

        inline bool Set(fbxReference* const &pValue ) { return Set(pValue, true); }

        bool Set(fbxReference* const &pValue, bool pCheckValueEquality )
        {
            KFbxObject* lValue = reinterpret_cast<KFbxObject*>(pValue);
            DisconnectAllSrcObject();
            if (lValue) {
                return ConnectSrcObject(lValue);
            }

            return false;
        }

        fbxReference* Get() const
        {
            KFbxObject* lValue = GetSrcObjectCount() > 0 ? GetSrcObject(0) : NULL;
            return reinterpret_cast<fbxReference*>(lValue);
        }

        friend class KFbxObject;
    };


    typedef KFbxTypedProperty<fbxBool1>         KFbxPropertyBool1;
    typedef KFbxTypedProperty<fbxInteger1>      KFbxPropertyInteger1;
    typedef KFbxTypedProperty<fbxDouble1>       KFbxPropertyDouble1;
    typedef KFbxTypedProperty<fbxDouble3>       KFbxPropertyDouble3;
    typedef KFbxTypedProperty<fbxDouble4>       KFbxPropertyDouble4;
    typedef KFbxTypedProperty<fbxString>        KFbxPropertyString;
    typedef KFbxTypedProperty<fbxReference*>    KFbxPropertyReference;

    enum eFbxConnectEventType {
        eFbxConnectRequest,
        eFbxConnect,
        eFbxConnected,
        eFbxDisconnectRequest,
        eFbxDisconnect,
        eFbxDisconnected
    };

    enum eFbxConnectEventDirection {
        eConnectEventSrc,
        eConnectEventDst
    };

    /** Class the handles Connection events.
      * \nosubgrouping
      */
    class KFBX_DLL KFbxConnectEvent
    {
        /**
          * \name Constructor and Destructors.
          */
        //@{
        public:
            inline KFbxConnectEvent(eFbxConnectEventType pType,eFbxConnectEventDirection pDir,KFbxProperty *pSrc,KFbxProperty *pDst)
                : mType(pType)
                , mDirection(pDir)
                , mSrc(pSrc)
                , mDst(pDst)
            {
            }
        //@}

        /**
          * \name Data Access.
          */
        //@{
        public:
            inline eFbxConnectEventType GetType() const { return mType; }
            inline eFbxConnectEventDirection GetDirection() const { return mDirection; }
            inline KFbxProperty &GetSrc()  const    { return *mSrc;  }
            inline KFbxProperty &GetDst()  const    { return *mDst;  }

            template < class T > inline T*  GetSrcIfObject(T const *pFBX_TYPE) const    { return mSrc->IsRoot() ? KFbxCast<T>(mSrc->GetFbxObject()) : (T*)0; }
            template < class T > inline T*  GetDstIfObject(T const *pFBX_TYPE) const    { return mDst->IsRoot() ? KFbxCast<T>(mDst->GetFbxObject()) : (T*)0; }
        //@}

        private:
            eFbxConnectEventType        mType;
            eFbxConnectEventDirection   mDirection;
            KFbxProperty*               mSrc;
            KFbxProperty*               mDst;
    };



#include <fbxfilesdk_nsend.h>

#endif // _KFbxProperty_h


