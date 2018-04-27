#ifndef __FBXEVENTS_FBXEVENTHANDLER_H__
#define __FBXEVENTS_FBXEVENTHANDLER_H__

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
#include <kfbxevents/kfbxevents.h>

// FBX include
#include <klib/kintrusivelist.h>

// FBX namespace begin
#include <fbxfilesdk_nsbegin.h>
namespace kfbxevents
{
    class KFbxListener;

    //-----------------------------------------------------------------
    class KFbxEventHandler
    {
    public:
        enum
        {
            eNODE_LISTENER = 0,
            eNODE_EMITTER,
            eNODE_COUNT
        };

        KFbxEventHandler(){}
        virtual ~KFbxEventHandler(){}

        // Handler handles a certain type of event
        virtual int GetHandlerEventType() = 0;
        virtual void FunctionCall(const KFbxEventBase& pEvent) = 0;
        virtual KFbxListener* GetListener() = 0;

        LISTNODE(KFbxEventHandler, eNODE_COUNT);
    };

    //-----------------------------------------------------------------
    template <typename EventType, typename ListenerType>
    class KFbxMemberFuncEventHandler : public KFbxEventHandler
    {
    // VC6Note: There's no reason why the callback is a (const EventType*) it is obvious that it should be
    //           a (const EventType&) but because of a VC6 template limitation, we put  a pointer. 
    typedef void (ListenerType::*CBFunction)(const EventType*);

    public:
        KFbxMemberFuncEventHandler(ListenerType* pListenerInstance, CBFunction pFunc) :
            mFunc(pFunc),
            mListener(pListenerInstance){}

        // From KFbxEventHandler
        virtual int GetHandlerEventType(){ return EventType::GetStaticTypeId(); }  
        virtual void FunctionCall(const KFbxEventBase& pEvent){ (*mListener.*mFunc)(reinterpret_cast<const EventType*>(&pEvent)); } 
        virtual KFbxListener* GetListener(){ return mListener;}
    
    private:
        ListenerType* mListener;

        // The callback function
        CBFunction mFunc;
    };

    //-----------------------------------------------------------------
    template <typename EventType, typename ListenerType>
    class KFbxConstMemberFuncEventHandler : public KFbxEventHandler
    {
    // VC6Note: There's no reason why the callback is a (const EventType*) it is obvious that it should be
    //           a (const EventType&) but because of a VC6 template limitation, we put  a pointer. 
    typedef void (ListenerType::*CBFunction)(const EventType*)const;

    public:
        KFbxConstMemberFuncEventHandler(ListenerType* pListenerInstance, CBFunction pFunc) :
            mFunc(pFunc),
            mListener(pListenerInstance){}

        // From KFbxEventHandler
        virtual int GetHandlerEventType(){ return EventType::GetStaticTypeId(); }    
        virtual void FunctionCall(const KFbxEventBase& pEvent){ (*mListener.*mFunc)(reinterpret_cast<const EventType*>(&pEvent)); }
        virtual KFbxListener* GetListener(){ return mListener;}

    private:
        ListenerType* mListener;

        // The callback function
        CBFunction mFunc;
    };

    //-----------------------------------------------------------------
    template <typename EventType>
    class KFbxFuncEventHandler : public KFbxEventHandler
    {
    // VC6Note: There's no reason why the callback is a (const EventType*,KFbxListener*) it is obvious that it should be
    //           a (const EventType&,KFbxListener*) but because of a VC6 template limitation, we put  a pointer.
    typedef void (*CBFunction)(const EventType*,KFbxListener*);

    public:
        KFbxFuncEventHandler(KFbxListener* pListener, CBFunction pFunc) :
            mFunc(pFunc),
            mListener(pListener){}

        // From KFbxEventHandler
        virtual int GetHandlerEventType(){ return EventType::GetStaticTypeId(); }   
        virtual void FunctionCall(const KFbxEventBase& pEvent){ (*mFunc)(reinterpret_cast<const EventType*>(&pEvent),mListener); }
        virtual KFbxListener* GetListener(){ return mListener; }

    private:
        KFbxListener* mListener;

        // The callback function
        CBFunction mFunc;
    };
}
// FBX namespace end
#include <fbxfilesdk_nsend.h>

#endif
