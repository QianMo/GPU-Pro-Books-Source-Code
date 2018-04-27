#ifndef _FBXSDK_PROPERTYMAP_H_
#define _FBXSDK_PROPERTYMAP_H_

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

#include <klib/kmap.h>
#include <kfbxplugins/kfbxproperty.h>
#include <kfbxplugins/kfbxobject.h>

#include <fbxfilesdk_nsbegin.h>

	template <class K,class T,class Compare> class KFBX_DLL KFbxMap
	{
		public:
			inline KFbxMap()
			{
			}

		private:
			typedef KMap<K,T,Compare>		KMapDef;
			KMapDef									mMap;
		public:
			typedef typename KMapDef::RecordType*	kIterator;

		public:
			inline void Add    (K const &pKey,T const &pType)
			{
				mMap.Insert( pKey,pType );
			}

			inline kIterator Find   (K const &pKey) const
			{
				return (kIterator)mMap.Find( pKey );
			}

			inline kIterator Find   (T const &pType) const
			{
			    kIterator lIterator = GetFirst();
				while (lIterator) {
					if (lIterator->GetValue()==pType) {
						return lIterator;
					}
					lIterator = GetNext(lIterator);
				}
				return 0;
			}

			inline void Remove (kIterator pIterator)
			{
				if (pIterator) mMap.Remove( pIterator->GetKey() );
			}

			inline kIterator GetFirst() const
			{
				return (kIterator)mMap.Minimum();
			}

			inline kIterator GetNext(kIterator pIterator) const
			{
				return (kIterator)pIterator ? pIterator->Successor() : 0;
			}

			inline void Clear() 
			{
				mMap.Clear();
			}

			inline void Reserve(int pSize)
			{
				mMap.Reserve( pSize );
			}

			inline int GetCount() const
			{
				return mMap.GetSize();
			}
	};

	/** This class maps types to properties
	  * \nosubgrouping
	  */
	template <class T,class Compare> class KFBX_DLL KFbxPropertyMap : public KFbxMap<T,KFbxProperty,Compare>
	{
		public:
			inline KFbxPropertyMap()
			{
			}

			inline KFbxProperty Get(typename KFbxMap<T,KFbxProperty,Compare>::kIterator pIterator)
			{
				return pIterator ? pIterator->GetValue() : KFbxProperty();
			}
	};

	/** This class compares strings.
	  * \nosubgrouping
	  * KFbxPropertyStringMap
	  */
    class KFbxMapKStringCompare {
		public:
	    inline int operator()(KString const &pKeyA, KString const &pKeyB) const
	    {
		    return (pKeyA < pKeyB) ? -1 : ((pKeyB < pKeyA) ? 1 : 0);
	    }
    };

	/** This class maps strings to properties
	  * \nosubgrouping
	  * KFbxObjectMap
	  */
	class KFBX_DLL KFbxPropertyStringMap : public KFbxPropertyMap<KString,KFbxMapKStringCompare>
	{
		public:
			inline KFbxPropertyStringMap()
			{
			}
	};

	/** This class maps types to objects.
	  * \nosubgrouping
	  * KFbxObjectMap
	  */
	template <class T,class Compare> class KFBX_DLL KFbxObjectMap : public KFbxMap<T,KFbxObject*,Compare>
	{
		public:
			inline KFbxObjectMap()
			{
			}
			inline KFbxObject* Get(typename KFbxMap<T,KFbxObject*,Compare>::kIterator pIterator)
			{
				return pIterator ? pIterator->GetValue() : 0;
			}
	};

	/** This class maps string names to objects.
	  * \nosubgrouping
	  * KFbxObjectStringMap
	  */
	class KFBX_DLL KFbxObjectStringMap : public KFbxObjectMap<class KString,class KFbxMapKStringCompare>
	{
		public:
			inline KFbxObjectStringMap()
			{
			}
	};

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_PROPERTYMAP_H_


