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
#include "CPUTControl.h"

// Constructor
//------------------------------------------------------------------------------
CPUTControl::CPUTControl():mControlVisible(true),
    mControlGraphicsDirty(false),
    mControlAutoArranged(true),
    mControlNeedsArrangmentResizing(true),
    mhotkey(KEY_NONE),
    mcontrolType(CPUT_CONTROL_UNKNOWN),
    mcontrolID(0),
    mpCallbackHandler(NULL),
    mControlState(CPUT_CONTROL_ACTIVE)
{
}

// Destructor
//------------------------------------------------------------------------------
CPUTControl::~CPUTControl()
{
}

// Control type/identifier routines that have a common implementation for all controls

// Sets the control's ID used for identification purposes (hopefully unique)
//------------------------------------------------------------------------------
void CPUTControl::SetControlID(CPUTControlID id)
{
    mcontrolID = id;
}

// Get the ID for this control
//------------------------------------------------------------------------------
CPUTControlID CPUTControl::GetControlID()
{
    return mcontrolID;
}

// Get the type of control this is (button/dropdown/etc)
//------------------------------------------------------------------------------
CPUTControlType CPUTControl::GetType()
{
    return mcontrolType;
}


// Set callback handler
//------------------------------------------------------------------------------
void CPUTControl::SetControlCallback(CPUTCallbackHandler *pHandler)
{
    mpCallbackHandler = pHandler;
}


// set whether controls is visible or not (it is still there, but not visible)
//------------------------------------------------------------------------------
void CPUTControl::SetVisibility(bool bVisible)
{
    mControlVisible = bVisible;
}

// visibility state
//------------------------------------------------------------------------------
bool CPUTControl::IsVisible()
{
    return mControlVisible;
}

// Set the hot key for keyboard events for this control
//------------------------------------------------------------------------------
void CPUTControl::SetHotkey(CPUTKey hotKey)
{
    mhotkey = hotKey;
}

// Get the hot key set for this control
//------------------------------------------------------------------------------
CPUTKey CPUTControl::GetHotkey()
{
    return mhotkey;
}

// Should this control be auto-arranged?
//------------------------------------------------------------------------------
void CPUTControl::SetAutoArranged(bool bIsAutoArranged)
{
    mControlAutoArranged = bIsAutoArranged;
}

//------------------------------------------------------------------------------
bool CPUTControl::IsAutoArranged()
{
    return mControlAutoArranged;
}

// Set the control to enabled or greyed out
//------------------------------------------------------------------------------
void CPUTControl::SetEnable(bool bEnabled)
{
    if(!bEnabled)
    {
        mControlState = CPUT_CONTROL_INACTIVE;
    }
    else
    {
        mControlState = CPUT_CONTROL_ACTIVE;
    }
}

// Return bool if the control is enabled/greyed out
//------------------------------------------------------------------------------
bool CPUTControl::IsEnabled()
{
    if(mControlState == CPUT_CONTROL_INACTIVE)
    {
        return false;
    }

    return true;
}