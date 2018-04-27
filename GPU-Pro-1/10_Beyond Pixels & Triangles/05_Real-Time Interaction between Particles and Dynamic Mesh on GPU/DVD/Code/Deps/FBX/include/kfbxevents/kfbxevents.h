#ifndef __FBXEVENTS_FBXEVENTS_H__
#define __FBXEVENTS_FBXEVENTS_H__

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

#include <kfbxplugins/kfbxtypes.h>
#include <kfbxplugins/kfbxdatatypes.h>
#include <kfbxmp/kfbxmutex.h>

// FBX namespace begin
#include <fbxfilesdk_nsbegin.h>
namespace kfbxevents
{
    /** FBX SDK event base class
	  * \nosubgrouping
	  */
    class KFBX_DLL KFbxEventBase
    {
      public:
		 /**
		   * \name Constructor and Destructor
		   */
	     //@{
		 //!Destructor
         virtual ~KFbxEventBase();
		 //@}

		 /** Retrieve the event type ID
		   * \return            type id
		   */
         virtual int GetTypeId() const = 0;

		 /** Force events to give us a name
		   * \return            event name 
		   */
         virtual const char* GetEventName() const = 0;   

		protected:
         static int GetStaticTypeId(char const*);

        private:
         static kfbxmp::KFbxMutex smMutex;
    };

    // Force events to declare a name by using an abstract method, and force them to use 
    // the proper name by making tne call from KFbxEvent<> go through the private static
    // method.
    #define KFBXEVENT_DECLARE(Class)                                                    \
      public: virtual const char* GetEventName() const { return FbxEventName(); }       \
      private: static const char* FbxEventName() { return #Class; }                     \
      friend class KFbxEvent<Class>;

    //
    // Similar to above, but to be used when you've got an event template, and the
    // type is something know to FBX
    //
    #define KFBXEVENT_DECLARE_FBXTYPE(Class, FBXType)                                  \
      public: virtual const char* GetEventName() const { return FbxEventName(); }      \
      private:                                                                         \
         static const char* FbxEventName() {                                           \
         static KString lEventName = KString(#Class) + KString("<") +                  \
         GetFbxDataType(FbxTypeOf(*((FBXType const*)0))).GetName() + ">";               \
                                                                                       \
         return lEventName.Buffer();                                                   \
      }                                                                                \
      friend class KFbxEvent< Class<FBXType> >;



    //This is for templates classes that will uses non fbxtypes in their templates
    //We force the the creation of an UNIQUE string for each types so that we can
    //retrieve the event within multiple DLLs

    //to be able to use this, the char EventName[] = "uniqueEventName"; must be declared
    //globally.

    #define KFBXEVENT_TEMPLATE_HEADER_NOT_FBXTYPE(ClassName, TemplateName)\
    template < class TemplateName, const char* T > \
    class ClassName: public kfbxevents:: KFbxEvent< ClassName <TemplateName,T> >\
    {\
        public: virtual const char* GetEventName() const {return FbxEventName();}\
        private: static const char* FbxEventName() {\
        static KString lEventName = (KString(#ClassName) +"<"+ KString(T) +">");\
        return lEventName.Buffer();\
        }\
        friend class KFbxEvent< ClassName<TemplateName, T> >;


    //This is the footer macro, to put at the end to close the template class
    //created by KFBXEVENT_TEMPLATE_HEADER_NOT_FBXTYPE
    #define KFBXEVENT_TEMPLATE_FOOTER_NOT_FBXTYPE()\
    };


    //---------------------------------------------------
    // EventT : We use the curiously recurring template pattern
    //          to initialize the typeId of each event type
    template<typename EventT>
    class KFbxEvent : public KFbxEventBase
    {
    public:
        virtual ~KFbxEvent(){}
        static void ForceTypeId(int pTypeId)
        {
            kfbxmp::KFbxMutexHelper lLock( smMutex );

            // This is to handle specific cases where the type ID must be hard coded
            // It is useful for shared event across DLL. We can then guarantee that
            // The ID of a certain type will always have the same ID
            smTypeId = pTypeId;
        }

        //! Note that this may be called from multiple threads.
        virtual int GetTypeId() const 
        {
			return GetStaticTypeId();
        }

        static int GetStaticTypeId() 
        {
            if( !smTypeId )
            {
                kfbxmp::KFbxMutexHelper lLock( smMutex );

                if( !smTypeId )
                {
                    // If this does not compile, you need to add 
                    // KFBXEVENT_DECLARE(YourEventClassName) to your class declaration
                    smTypeId  = KFbxEventBase::GetStaticTypeId(EventT::FbxEventName());
                }
            }

           return smTypeId;
        }

    private:
        static int smTypeId;
        static kfbxmp::KFbxMutex smMutex;
    };

    // Static members implementation
    template<typename EventT>
    int KFbxEvent<EventT>::smTypeId = 0;
    template<typename EventT>
    kfbxmp::KFbxMutex KFbxEvent<EventT>::smMutex;
}
using namespace kfbxevents;


// FBX namespace end
#include <fbxfilesdk_nsend.h>

#endif
