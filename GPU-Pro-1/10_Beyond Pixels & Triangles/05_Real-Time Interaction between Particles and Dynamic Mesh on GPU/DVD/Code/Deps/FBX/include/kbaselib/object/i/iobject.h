/*!  \file iobject.h
 */

#ifndef _FBXSDK_IOBJECT_H_
#define _FBXSDK_IOBJECT_H_
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
#include <kbaselib_h.h>

#include <kbaselib_nsbegin.h>

    #ifdef K_PLUGIN
	    #define IObject    IKObject
	    #define HIObject   HIKObject
	    #define IObjectID  IKObjectID
	    #define IKObjectID 0
	    #define IObjectDerived      public IKObject
    #else
	    #define IObjectID  0
	    #define IObjectDerived      public IObject
    #endif

    /** forwarding **/
    K_FORWARD( IObject )

    /** other types **/
    typedef unsigned long kInterfaceID;

    /** MACRO Base definitions **/
    #define FncDefine(PFNC,ISPURE) virtual PFNC##ISPURE
    #define Pure           =0;
    #define Implementation ;
    #define IObjectImplement(ClassName,IObjectOwner) \
        HIObject ClassName##::IQuery(kInterfaceID pInterfaceID,int IsLocal) {\
            if ((IObjectOwner!=NULL) && (!IsLocal)) {\
                return IObjectOwner->IQuery(pInterfaceID);\
            }\
            switch (pInterfaceID) 
    #define IObjectEnd return NULL; }
    #define ILOCAL          1

    #define IQUERY(Object,Interface) ((H##Interface)((Object)->IQuery(Interface##ID,0)))
    #define IQUERYLOCAL(Object,Interface) ( (H##Interface)((Object)->IQuery(Interface##ID,ILOCAL)))

    #define IQ(Object,Interface) IQUERY(Object,Interface)
    #define IQL(Object,Interface) IQUERYLOCAL(Object,Interface)

    /*********************************************************************
    *  Class IObject
    *    This is the base class of all Important interfaces of the system 
    *********************************************************************/

    #define IObject_Declare(IsPure)\
    public:\
        virtual HIObject	IQuery(kInterfaceID pInterfaceID, int IsLocal=0)IsPure\
        virtual void		Destroy(int IsLocal=0)IsPure\

    #define IQuery_Declare(IsPure)\
    public:\
        virtual HIObject IQuery (kInterfaceID pInterfaceID, int IsLocal=0)IsPure\
        
    class IObject {
        IObject_Declare(Pure)
    };

    typedef HIObject (* kObjectCreatorFnc)(HIObject pOwner,char *pName,void *pData);

    #ifdef KARCH_DEV_MSC
	    #define K_INTERFACE_SPECIAL __declspec(novtable)
	    #define NO_DLL				__declspec()		// Use for the Module parameter if the interface is not exported
    #else 
	    #define K_INTERFACE_SPECIAL 
	    #define NO_DLL									// Use for the Module parameter if the interface is not exported
    #endif

    #define K_INTERFACE( Name,Id ) \
    K_FORWARD( Name ) \
    const int Name##ID = Id; \
    class K_INTERFACE_SPECIAL Name : IObjectDerived { \
        Name##_Declare(Pure) \
    } 

#include <kbaselib_nsend.h>

#endif  // Must be the last line of the include file


