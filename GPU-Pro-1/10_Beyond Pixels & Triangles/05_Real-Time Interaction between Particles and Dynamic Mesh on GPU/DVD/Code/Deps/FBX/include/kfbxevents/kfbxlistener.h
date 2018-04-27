#ifndef __FBXEVENTS_FBXLISTENERS_H__
#define __FBXEVENTS_FBXLISTENERS_H__

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

// Local includes
#include <kfbxevents/kfbxemitter.h>
#include <kfbxevents/kfbxeventhandler.h>

// FBX include
#include <klib/kintrusivelist.h>

// FBX namespace begin
#include <fbxfilesdk_nsbegin.h>

namespace kfbxevents
{
    /**FBX SDK listener class
	  * \nosubgrouping
	  */
    class KFBX_DLL KFbxListener
    {
    public:
		/**
		  * \name Constructor and Destructor
		  */
		//@{
		//!Destructor.
        ~KFbxListener();
		//!Constructor.
        KFbxListener(){}
		//@}
       
		////////////////////////////////////////////////////////////////////////////////////////
        /**Bind an  Event handler.
		  * \param pEmitter          Event emitter.
		  * \param pFunc             The call back function.
		  * \return                  the event handler binded.
		  */

        template <typename EventType,typename ListenerType>
        KFbxEventHandler* Bind(KFbxEmitter& pEmitter, void (ListenerType::*pFunc)(const EventType*))

        {
            KFbxMemberFuncEventHandler<EventType,ListenerType>* eventHandler = 
                new KFbxMemberFuncEventHandler<EventType,ListenerType>(static_cast<ListenerType*>(this),pFunc);
            pEmitter.AddListener(*eventHandler);
            mEventHandler.PushBack(*eventHandler);
            return eventHandler;
        }

        ////////////////////////////////////////////////////////////////////////////////////////
		/**Bind an  Event handler.
		  * \param pEmitter          Event emitter.
		  * \param pFunc             The call back function.
		  * \return                  the event handler binded.
		  */
        template <typename EventType,typename ListenerType>
        KFbxEventHandler* Bind(KFbxEmitter& pEmitter, void (ListenerType::*pFunc)(const EventType*)const)
        {
            KFbxConstMemberFuncEventHandler<EventType,ListenerType>* eventHandler = 
                        new KFbxConstMemberFuncEventHandler<EventType,ListenerType>(static_cast<ListenerType*>(this),pFunc);
            pEmitter.AddListener(*eventHandler);
            mEventHandler.PushBack(*eventHandler);
            return eventHandler;
        }

        ////////////////////////////////////////////////////////////////////////////////////////
		/**Bind an  Event handler.
		  * \param pEmitter          Event emitter.
		  * \param pFunc             The call back function.
		  * \return                  the event handler binded.
		  */
        template <typename EventType>
        KFbxEventHandler* Bind(KFbxEmitter& pEmitter, void (*pFunc)(const EventType*,KFbxListener*))
        {
            KFbxFuncEventHandler<EventType>* eventHandler = 
                            new KFbxFuncEventHandler<EventType>(this, pFunc);
            pEmitter.AddListener(*eventHandler);
            mEventHandler.PushBack(*eventHandler);
            return eventHandler;
        }
        
		/**Unbind an event handler.
		  * \param aBindId                 The event handler to unbind.
		  */
        void Unbind(const KFbxEventHandler* aBindId);

    private:
        typedef KIntrusiveList<KFbxEventHandler, KFbxEventHandler::eNODE_LISTENER> EventHandlerList;
        EventHandlerList mEventHandler;
    };
}

// FBX namespace end
#include <fbxfilesdk_nsend.h>

#endif
