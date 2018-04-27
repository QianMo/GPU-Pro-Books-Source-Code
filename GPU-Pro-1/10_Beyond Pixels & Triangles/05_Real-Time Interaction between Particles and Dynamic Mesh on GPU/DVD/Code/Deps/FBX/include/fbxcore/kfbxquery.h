#ifndef _FBXSDK_QUERY_H_
#define _FBXSDK_QUERY_H_

/**************************************************************************************

 Copyright ?2001 - 2008 Autodesk, Inc. and/or its licensors.
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

#include <fbxcore/kfbxpropertydef.h>
#include <kfbxplugins/kfbxproperty.h>

#include <fbxfilesdk_nsbegin.h>


    /***********************************************
      KFbxQuery
    ************************************************/

/**	\brief Class to manage query.
* \nosubgrouping
*/
    class KFBX_DLL KFbxQuery {
        // Overridable Test functions
        public:
			//! Get unique filter Id
            virtual kFbxFilterId                    GetUniqueId() const     { return 0x1452e230; }
            
			//! Judge if the given property is valid.
			virtual bool IsValid(KFbxProperty const &pProperty) const;
			//! Judge if the given property and connection type are valid.
            virtual bool IsValid(KFbxProperty const &pProperty,kFbxConnectionType pType) const;

            /**This compares whether two KFbxQuery are the same, NOT whether the query
              * matches or not.  It's strictly the equivalent of an operator==, but virtual.
			  * \param pOtherQuery The given KFbxQuery
             */
            virtual bool IsEqual(KFbxQuery *pOtherQuery)    const;

        public:
			//! Add one to ref count .
            void                                    Ref();
			//! Minus one to ref count, if ref count is zero, delete this query object.
            void                                    Unref();
        protected:
            KFbxQuery();
            virtual ~KFbxQuery();
        private:
            class KFbxInternalFilter : public KFbxConnectionPointFilter{
                public:
                    KFbxInternalFilter(KFbxQuery *pQuery);
                    ~KFbxInternalFilter();

                // internal functions
                public:
                    KFbxConnectionPointFilter*      Ref();
                    void                            Unref();
                    kFbxFilterId                    GetUniqueId() const { return mQuery->GetUniqueId(); }
                    bool                            IsValid             (KFbxConnectionPoint*   pConnect) const;
                    bool                            IsValidConnection   (KFbxConnectionPoint*   pConnect,kFbxConnectionType pType) const;
                    bool                            IsEqual             (KFbxConnectionPointFilter* pConnectFilter) const;
                public:
                    KFbxQuery*      mQuery;

            };

            KFbxInternalFilter  mFilter;
            int                 mRefCount;

            friend class KFbxProperty;
            friend class KFbxInternalFilter;
        };

    /***********************************************
      KFbxQueryOperator (binary operators)
    ************************************************/
    enum eFbxQueryOperator { eFbxAnd,eFbxOr } ;

	/**	\brief Class to manage query operator.
	* \nosubgrouping
	*/

    class KFBX_DLL KFbxQueryOperator : public KFbxQuery {
        //
        public:
			//! Create new query operator.
            static KFbxQueryOperator* Create(KFbxQuery* pA,eFbxQueryOperator pOperator,KFbxQuery* pB);

        protected:
            KFbxQueryOperator(KFbxQuery* pA,eFbxQueryOperator pOperator,KFbxQuery* pB);
            virtual ~KFbxQueryOperator();

        // Test functions
        public:

			//! Get unique filter Id
            virtual kFbxFilterId    GetUniqueId() const{ return 0x945874; }

			//! Test if the given property is valid for this query operator.
            virtual bool IsValid(KFbxProperty const &pProperty) const;

			//! Test if the given property and connection type are valid for this query operator.
            virtual bool IsValid(KFbxProperty const &pProperty,kFbxConnectionType pType) const;
			
			//! Test if this query operator is equal with the given query operator.
            virtual bool IsEqual(KFbxQuery *pOtherQuery)    const;
        private:
            KFbxQuery           *mA,*mB;
            eFbxQueryOperator   mOperator;
    };

    /***********************************************
      KFbxUnaryQueryOperator
    ************************************************/
    enum eFbxUnaryQueryOperator { eFbxNot };

	/**	\brief Class to manage unary query operator.
	* \nosubgrouping
	*/
    class KFBX_DLL KFbxUnaryQueryOperator : public KFbxQuery {
        //
        public:
			//! Create new unary query operator.
            static KFbxUnaryQueryOperator* Create(KFbxQuery* pA,eFbxUnaryQueryOperator pOperator);

        protected:
            KFbxUnaryQueryOperator(KFbxQuery* pA,eFbxUnaryQueryOperator pOperator);
            virtual ~KFbxUnaryQueryOperator();

        // Test functions
        public:

			//! Get unique filter Id
            virtual kFbxFilterId    GetUniqueId() const{ return 0x945874BB; }

			//! Test if the given property is valid for this unary query operator.
            virtual bool IsValid(KFbxProperty const &pProperty) const;

            //! Test if the given property and connection type are valid for this unary query operator.
            virtual bool IsValid(KFbxProperty const &pProperty,kFbxConnectionType pType) const;

			//! Test if this unary query operator is equal with the given unary query operator.
            virtual bool IsEqual(KFbxQuery *pOtherQuery)    const;
        private:
            KFbxQuery                *mA;
            eFbxUnaryQueryOperator   mOperator;
    };

    /***********************************************
      KFbxQueryClassId -- match anywhere in the hierarchy of an object.
    ************************************************/
	/**	\brief Class to manage query class Id.
	* \nosubgrouping
	*/
    class KFBX_DLL KFbxQueryClassId : public KFbxQuery {
        //
        public:
			//! Creat a new query class Id.
            static KFbxQueryClassId* Create(kFbxClassId pClassId);

        protected:
            KFbxQueryClassId(kFbxClassId pClassId);

        // Test functions
        public:

			//! Get unique filter Id
            virtual kFbxFilterId    GetUniqueId() const{ return 0x14572785; }

			//! Test if the given property is valid for this query class Id.
            virtual bool IsValid(KFbxProperty const &pProperty) const;
			//! Test if the given property is valid for this query class Id.
            virtual bool IsValid(KFbxProperty const &pProperty,kFbxConnectionType pType) const;

            //! Test if this query class Id is equal with the given query class Id.
            virtual bool IsEqual(KFbxQuery *pOtherQuery)    const;
        private:
            kFbxClassId             mClassId;
    };

    /***********************************************
      KFbxQueryIsA -- Exact match.
    ************************************************/

	/**	\brief Class to manage query property .
	* \nosubgrouping
	*/
    class KFBX_DLL KFbxQueryIsA : public KFbxQuery {
        //
        public:
			//! Create a new query IsA object
            static KFbxQueryIsA* Create(kFbxClassId pClassId);

        protected:
            KFbxQueryIsA(kFbxClassId pClassId);

        // Test functions
        public:

			//! Get unique filter Id
            virtual kFbxFilterId    GetUniqueId() const{ return 0x1457278F; }

			//! Test if the given property is valid for this query IsA.
            virtual bool IsValid(KFbxProperty const &pProperty) const;

			//! Test if the given property is valid for this query IsA.
            virtual bool IsValid(KFbxProperty const &pProperty,kFbxConnectionType pType) const;

			//! Test if this query is equal with the given query .
            virtual bool IsEqual(KFbxQuery *pOtherQuery)    const;
        private:
            kFbxClassId             mClassId;
    };

    /***********************************************
      KFbxQueryProperty
    ************************************************/

	/**	\brief Class to manage query property .
	* \nosubgrouping
	*/
    class KFBX_DLL KFbxQueryProperty : public KFbxQuery {
        //
        public:
			//! Create new query property
            static KFbxQueryProperty* Create();

        protected:
            KFbxQueryProperty();

        // Test functions
        public:

			//! Get unique filter Id
            virtual kFbxFilterId    GetUniqueId() const{ return 0x9348203; }

			//! Test if this query for given property is valid.
            virtual bool IsValid(KFbxProperty const &pProperty) const;

			//! Test if this query for given property is valid.
            virtual bool IsValid(KFbxProperty const &pProperty,kFbxConnectionType pType) const;
			//! Return true.
            virtual bool IsEqual(KFbxQuery *pOtherQuery)    const;
        private:
    };

    /***********************************************
      KFbxQueryConnectionType
    ************************************************/

	/**	\brief Class to manage query connection type.
	* \nosubgrouping
	*/
    class KFBX_DLL KFbxQueryConnectionType : public KFbxQuery {
        //
        public:

			//! Create a new query connection type
            static KFbxQueryConnectionType* Create(kFbxConnectionType pConnectionType);

        protected:
            KFbxQueryConnectionType(kFbxConnectionType pConnectionType);

        // Test functions
        public:

			//! Get unique filter Id
            virtual kFbxFilterId    GetUniqueId() const{ return 0x14587275; }

			//! Return true.
            virtual bool IsValid(KFbxProperty const &pProperty) const;
			//! Test if the given connection type is valid.
            virtual bool IsValid(KFbxProperty const &pProperty,kFbxConnectionType pType) const;
			//! Test if this query connection type is equal with the given query connection type.
            virtual bool IsEqual(KFbxQuery *pOtherQuery)    const;
        private:
            kFbxConnectionType              mConnectionType;
    };

    /***********************************************
      KFbxCriteria
    ************************************************/
    class KFbxCriteria {
        public:
            static KFbxCriteria ConnectionType(kFbxConnectionType pConnectionType)
            {
                return KFbxCriteria(KFbxQueryConnectionType::Create(pConnectionType));
            }
            static KFbxCriteria ObjectType(kFbxClassId pClassId)       // Hierarchy match
            {
                return KFbxCriteria(KFbxQueryClassId::Create(pClassId));
            }
            static KFbxCriteria ObjectIsA(kFbxClassId pClassId)        // Exact match
            {
                return KFbxCriteria(KFbxQueryIsA::Create(pClassId));
            }
            static KFbxCriteria Property()
            {
                return KFbxCriteria(KFbxQueryProperty::Create());
            }
            inline KFbxCriteria()
                : mQuery(0)
            {
            }
            inline KFbxCriteria(KFbxCriteria const &pCriteria)
                : mQuery(pCriteria.mQuery)
            {
                if( mQuery )
                    mQuery->Ref();
            }

            inline ~KFbxCriteria()
            {
                if( mQuery )
                    mQuery->Unref();
            }
        private:
            inline KFbxCriteria(KFbxQuery* pQuery)
                : mQuery(pQuery)
            {
            }
        public:
            inline KFbxCriteria &operator=(KFbxCriteria const &pCriteria)
            {
                if( this != &pCriteria )
                {
                    KFbxQuery* lQuery = mQuery;
                    mQuery = pCriteria.mQuery;

                    if( mQuery )
                        mQuery->Ref();

                    if( lQuery )
                        lQuery->Unref();
                }

                return *this;
            }
            inline KFbxCriteria operator && (KFbxCriteria const &pCriteria) const
            {
                return KFbxCriteria(KFbxQueryOperator::Create(GetQuery(),eFbxAnd,pCriteria.GetQuery()));
            }
            inline KFbxCriteria operator || (KFbxCriteria const &pCriteria) const
            {
                return KFbxCriteria(KFbxQueryOperator::Create(GetQuery(),eFbxOr,pCriteria.GetQuery()));
            }
            inline KFbxCriteria operator !() const
            {
                return KFbxCriteria(KFbxUnaryQueryOperator::Create(GetQuery(), eFbxNot));
            }

        public:
            inline KFbxQuery*   GetQuery() const { return mQuery; }
        private:
            KFbxQuery*  mQuery;
    };

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_Document_H_


