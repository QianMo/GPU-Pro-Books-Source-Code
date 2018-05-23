//--------------------------------------------------------------------------------------
// Copyright 2013 Intel Corporation
// All Rights Reserved
//
// Permission is granted to use, copy, distribute and prepare derivative works of this
// software for any purpose and without fee, provided, that the above copyright notice
// and this statement appear in all copies.  Intel makes no representations about the
// suitability of this software for any purpose.  THIS SOFTWARE IS PROVIDED "AS IS."
// INTEL SPECIFICALLY DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, AND ALL LIABILITY,
// INCLUDING CONSEQUENTIAL AND OTHER INDIRECT DAMAGES, FOR THE USE OF THIS SOFTWARE,
// INCLUDING LIABILITY FOR INFRINGEMENT OF ANY PROPRIETARY RIGHTS, AND INCLUDING THE
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  Intel does not
// assume any responsibility for any errors which may appear in this software nor any
// responsibility to update it.
//--------------------------------------------------------------------------------------
#ifndef __CPUTCALLBACKHANDLER_H__
#define __CPUTCALLBACKHANDLER_H__

#include "CPUT.h"
#include "CPUTControl.h"

//forward declarations
class CPUTControl;
typedef unsigned int UINT;
typedef UINT CPUTControlID;
typedef UINT CPUTEventID;
#define UNREFERENCED_PARAMETER(P) (P)

class CPUTCallbackHandler
{
public:
    virtual void HandleCallbackEvent( CPUTEventID Event, CPUTControlID ControlID, CPUTControl *pControl ) {UNREFERENCED_PARAMETER(Event);UNREFERENCED_PARAMETER(ControlID);UNREFERENCED_PARAMETER(pControl);}
};


#endif // #ifndef __CPUTCALLBACKHANDLER_H__
