/*!  \file kbaselib_forward.h
 */

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

#ifndef KBASELIB_FORWARD_H
#define KBASELIB_FORWARD_H

#include <kbaselib_h.h>

K_FORWARD(IError)

#include <kbaselib_nsbegin.h> // namespace 

    class KgeTVector;
    class KgeRVector;
    class KgeSVector;
    class KgeRGBVector;
    class KgeRGBAVector;
    class KgeRMatrix;
    class KgeQuaternion;
    class KgeAMatrix;
    
    typedef KgeTVector KgeVector;
    
    struct KMergeInfo;

    // Forward
    K_FORWARD(KBitSet)
    K_FORWARD(KTime)

    K_FORWARD(KMpMutex)
    K_FORWARD(KMpFastMutex)
    K_FORWARD(KMpTrigger)
    K_FORWARD(KMpGate)
    K_FORWARD(KMpFastLock)
    K_FORWARD(KMpThread)
    K_FORWARD(KMpFifo)
    K_FORWARD(KMpStack)

    K_FORWARD(KError)
    K_FORWARD(KObject)
    K_FORWARD(KProperty)
    K_FORWARD(HdlKObject)
    K_FORWARD(KStateMember)
    K_FORWARD(IRegister)
    K_FORWARD(KPlug)
    K_FORWARD(KEvaluateInfo)
    K_FORWARD(KEventEntity)
    K_FORWARD(KEvaluationState)
    K_FORWARD(IApplyManager)
    K_FORWARD(IMergeManager)
    K_FORWARD(KDataType)
    K_FORWARD(IKObject)

    K_FORWARD(KRenamingStrategy)    
    K_FORWARD(KObjectList)
    K_FORWARD(KMBTransform)

    K_FORWARD(KChainedFile)

    K_FORWARD(KFbxField)
    K_FORWARD(KFbxFieldList)
    K_FORWARD(KUniqueNameObjectList)
    K_FORWARD(KConfigFile)
    K_FORWARD(KFbx)

    K_FORWARD(KSet)    
    
    K_FORWARD(Ktcpip)

    K_FORWARD(KStringList)

    K_FORWARD(KCharPtrSet)
    
#include <kbaselib_nsend.h> // namespace 

#endif
