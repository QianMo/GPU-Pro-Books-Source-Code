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
#include "CPUTGuiController.h"

// Constructor
//-----------------------------------------------------------------------------
CPUTGuiController::CPUTGuiController():
    mControlPanelIDList(NULL),
    mActiveControlPanelSlotID(CPUT_CONTROL_ID_INVALID),
    mpHandler(NULL),
    mpFocusControl(NULL),
    mbAutoLayout(true),
    mResourceDirectory(_L("./")),
    mUberBufferDirty(true),
    mRecalculateLayout(false)
{
    // clear all controls and panel lists
    DeleteAllControls();
}

// Destructor
//-----------------------------------------------------------------------------
CPUTGuiController::~CPUTGuiController()
{
    // cleanup
    DeleteAllControls();
}

//CPUTEventHandler members

// Handle mouse events
// The gui controller dispatches the event to each control until handled or
// it passes through unhandled
//-----------------------------------------------------------------------------
CPUTEventHandledCode CPUTGuiController::HandleMouseEvent(int x, int y, int wheel, CPUTMouseState state)
{
    // if there is no active panel, then we just return
    if(CPUT_CONTROL_ID_INVALID == mActiveControlPanelSlotID)
    {
        return CPUT_EVENT_PASSTHROUGH;
    }

    // walk the list of controls on the screen and see if they are to handle any of these events
    CPUTEventHandledCode EventResult = CPUT_EVENT_PASSTHROUGH;
    for(UINT i=0; i<mControlPanelIDList[mActiveControlPanelSlotID]->mControlList.size(); i++)
    {
        if(CPUT_EVENT_HANDLED == EventResult)
        {
            // walk the rest of the controls, updating them with the event, but with an 'invalid' location
            // this is a not-so-great but works bug fix for this problem:
            // If you click on another control besides a dropped-down dropdown, then you get a painting error
            // You need to send a closed event to any remaining dropdowns...
            mControlPanelIDList[mActiveControlPanelSlotID]->mControlList[i]->HandleMouseEvent(-1,-1,wheel,state);
        }
        else
        {
            EventResult = mControlPanelIDList[mActiveControlPanelSlotID]->mControlList[i]->HandleMouseEvent(x,y,wheel,state);

            // if the control says it handled this event, do not pass it through to underlying controls
            // this is important for things like dropdowns that could dynamically overlap other controls
            if( CPUT_EVENT_HANDLED == EventResult)
            {
                mpFocusControl = mControlPanelIDList[mActiveControlPanelSlotID]->mControlList[i];
                //return EventResult;
            }
        }
    }

    return EventResult; //CPUT_EVENT_PASSTHROUGH;
}

// Turns on/off automatic arrangement of controls on right side of screen
//-----------------------------------------------------------------------------
void CPUTGuiController::EnableAutoLayout(bool UseAutoLayout)
{
    mbAutoLayout = UseAutoLayout;
}

// sets the resource directory to use when loading GUI resources
//-----------------------------------------------------------------------------
CPUTResult CPUTGuiController::SetResourceDirectory(const cString ResourceDirectory)
{
    // check to see if the specified directory is valid
    CPUTResult result = CPUT_SUCCESS;

    // resolve the directory to a full path
    cString FullPath;
    CPUTOSServices *pServices = CPUTOSServices::GetOSServices();
    result = pServices->ResolveAbsolutePathAndFilename(ResourceDirectory, &FullPath);
    if(CPUTFAILED(result))
    {
        return result;
    }

    // check existence of directory
    result = pServices->DoesDirectoryExist(FullPath);
    if(CPUTFAILED(result))
    {
        return result;
    }

    // set the resource directory (absolute path)
    mResourceDirectory = FullPath;

    return result;
}

// gets the resource directory used when loading GUI resources
//-----------------------------------------------------------------------------
void CPUTGuiController::GetResourceDirectory(cString &ResourceDirectory)
{
    ResourceDirectory = mResourceDirectory;
}



// Panels

// finds panel with matching ID and sets it as the active one
// if panelID is CPUT_CONTROL_ID_INVALID - it will disable all panels
//-----------------------------------------------------------------------------
CPUTResult CPUTGuiController::SetActivePanel(CPUTControlID panelID)
{
    CPUTControlID panelSlotID = FindPanelIDIndex(panelID);

    // if we found it, set it active
    if(CPUT_CONTROL_ID_INVALID == panelSlotID)
    {
        return CPUT_ERROR_NOT_FOUND;
    }

    // store previously active control
    mControlPanelIDList[mActiveControlPanelSlotID]->mpFocusControl = mpFocusControl;

    // change the active panel and refresh screen
    mActiveControlPanelSlotID = panelSlotID;
    mpFocusControl = mControlPanelIDList[mActiveControlPanelSlotID]->mpFocusControl;

    // trigger refresh
    if(mbAutoLayout)
    {
        Resize();
    }

    return CPUT_SUCCESS;
}

// returns the ID of the active panel
//-----------------------------------------------------------------------------
CPUTControlID CPUTGuiController::GetActivePanelID()
{
    if(CPUT_CONTROL_ID_INVALID == mActiveControlPanelSlotID)
        return CPUT_CONTROL_ID_INVALID;

    return mControlPanelIDList[mActiveControlPanelSlotID]->mpanelID;
}

// Get the list of controls currently being displayed
//-----------------------------------------------------------------------------
CPUTResult CPUTGuiController::GetActiveControlList(std::vector<CPUTControl*> *ppControlList)
{
	ASSERT(ppControlList != NULL, _L("CPUTGuiController::GetActiveControlList - pControlList is NULL"));

    if(CPUT_CONTROL_ID_INVALID == mActiveControlPanelSlotID)
    {
        // return CPUT_GUI_INVALID_CONTROL_ID;
    }

    if(NULL==ppControlList)
    {
        return CPUT_ERROR_INVALID_PARAMETER;
    }

    // todo: make a copy instead to avoid deletion problems?
    *ppControlList = mControlPanelIDList[mActiveControlPanelSlotID]->mControlList;

    return CPUT_SUCCESS;
}


// removes specified control from the panel (does not delete the control)
//-----------------------------------------------------------------------------
CPUTResult CPUTGuiController::RemoveControlFromPanel(CPUTControlID controlID, CPUTControlID panelID)
{
    CPUTControlID panelSlotID;
    if(CPUT_CONTROL_ID_INVALID==panelID)
    {
        // use the currently active panel if none specified
        panelSlotID = mActiveControlPanelSlotID;
    }
    else
    {
        panelSlotID = FindPanelIDIndex(panelID);
    }

    // invalid panel
    //if(CPUT_CONTROL_ID_INVALID == panelSlotID)
        //return CPUT_ERROR_INVALID_PARAMETER;

    // walk list of controls in the panel and see if control is there
    for(UINT i=0; i<mControlPanelIDList[panelSlotID]->mControlList.size(); i++)
    {
        if( controlID == mControlPanelIDList[panelSlotID]->mControlList[i]->GetControlID() )
        {
            mControlPanelIDList[panelSlotID]->mControlList.erase( (mControlPanelIDList[panelSlotID]->mControlList.begin() + i) );

            // trigger refresh
            if(mbAutoLayout)
            {
                Resize();
            }

            return CPUT_SUCCESS;
        }
    }
    return CPUT_WARNING_NOT_FOUND;
}

// removes panel and deletes all controls associated with it
//-----------------------------------------------------------------------------
CPUTResult CPUTGuiController::DeletePanel(CPUTControlID panelID)
{
    // find the panel they specified
    CPUTControlID panelSlotID = FindPanelIDIndex(panelID);

    // panel not found
    //if(CPUT_CONTROL_ID_INVALID == panelSlotID)
        //return CPUT_ERROR_INVALID_PARAMETER;

    // walk the panel and delete all the controls in it
    for(UINT i=0; i<mControlPanelIDList[panelSlotID]->mControlList.size(); i++)
    {
        // delete each control
        SAFE_DELETE_ARRAY(mControlPanelIDList[panelSlotID]->mControlList[i]);        
    }

    // remove this panel from the control list
    mControlPanelIDList.erase(mControlPanelIDList.begin()+panelSlotID);

    // if the panel you delete is the active one, set the active panel to first
    // or invalid if none are left
    if(mActiveControlPanelSlotID == panelSlotID)
    {
        // set panel to the first panel
        if(0 == mControlPanelIDList.size())
            mActiveControlPanelSlotID = CPUT_CONTROL_ID_INVALID;
        else
            mActiveControlPanelSlotID = 0;
    }


    // trigger refresh
    if(mbAutoLayout)
    {
        Resize();
    }

    return CPUT_SUCCESS;
}

// private: Finds the index of the specified panel ID code in mControlPanelIDList[]
//-----------------------------------------------------------------------------
UINT CPUTGuiController::FindPanelIDIndex(CPUTControlID panelID)
{
    CPUTControlID foundID = CPUT_CONTROL_ID_INVALID;

    for(UINT i=0; i<mControlPanelIDList.size(); i++)
    {
        if(panelID == mControlPanelIDList[i]->mpanelID)
            return i;
    }

    return foundID;
}

// Returns the number of controls in the currently ACTIVE panel
//-----------------------------------------------------------------------------
int CPUTGuiController::GetNumberOfControlsInPanel(CPUTControlID panelID)
{
    // if not specified, returns count of currently active pane
    if(-1 == panelID)
    {
        if(CPUT_CONTROL_ID_INVALID == mActiveControlPanelSlotID)
            return 0;
        return (int) mControlPanelIDList[mActiveControlPanelSlotID]->mControlList.size();
    }

    // if panelID specified, return that number, or 0 if not found
    UINT foundID = FindPanelIDIndex(panelID);
    if(CPUT_CONTROL_ID_INVALID != foundID)
        return (int) mControlPanelIDList[foundID]->mControlList.size();

    return CPUT_CONTROL_ID_INVALID;
}

// is the control in the panel?
//-----------------------------------------------------------------------------
bool CPUTGuiController::IsControlInPanel(CPUTControlID controlID, CPUTControlID panelID)
{
    CPUTControlID panelSlotID;
    if(-1==panelID)
    {
        // use the currently active panel if none specified
        panelSlotID = mActiveControlPanelSlotID;
    }
    else
    {
        panelSlotID = FindPanelIDIndex(panelID);
    }

    // invalid panel
    if(CPUT_CONTROL_ID_INVALID == panelSlotID)
        return false;

    // walk list of controls in the panel and see if control is there
    for(UINT i=0; i<mControlPanelIDList[panelSlotID]->mControlList.size(); i++)
    {
        if( controlID == mControlPanelIDList[panelSlotID]->mControlList[i]->GetControlID() )
            return true;
    }
    return false;
}

// Control management

// Add a control (to a panel)
//-----------------------------------------------------------------------------
CPUTResult CPUTGuiController::AddControl(CPUTControl *pControl, CPUTControlID panelID)
{
    //if(NULL == pControl)
        //return CPUT_ERROR_INVALID_PARAMETER;

    // set the global callback handler for this object
    pControl->SetControlCallback(mpHandler);

    CPUTControlID panelSlotID = FindPanelIDIndex(panelID);

    // if the panel wasn't found, add a new one
    if(CPUT_CONTROL_ID_INVALID == panelSlotID)
    {
        Panel *pNewControlPanel = new Panel();
        pNewControlPanel->mpanelID = panelID;
        pNewControlPanel->mControlList.clear();
        pNewControlPanel->mpFocusControl = NULL;

        mControlPanelIDList.push_back( pNewControlPanel );
        panelSlotID = (int)mControlPanelIDList.size()-1;

        // make the newly added panel active if none was
        // active before
        if(CPUT_CONTROL_ID_INVALID == mActiveControlPanelSlotID)
            mActiveControlPanelSlotID = panelSlotID;
    }

    // store the control in the list
    mControlPanelIDList[panelSlotID]->mControlList.push_back(pControl);

    // trigger a resize to position controls optimally
    if(mbAutoLayout)
    {
        Resize();
    }
    return CPUT_SUCCESS;
}

// returns a pointer to the specified control
//-----------------------------------------------------------------------------
CPUTControl* CPUTGuiController::GetControl(CPUTControlID controlID, CPUTResult *pResult)
{
	if (pResult)
    {
		*pResult = CPUT_SUCCESS;
    }

    for(UINT i=0; i<mControlPanelIDList.size(); i++)
    {
        for(UINT j=0; j<mControlPanelIDList[i]->mControlList.size(); j++)
        {
            if(controlID == mControlPanelIDList[i]->mControlList[j]->GetControlID())
            {
                return mControlPanelIDList[i]->mControlList[j];
            }
        }
    }
	
	if (pResult)
    {
		*pResult = CPUT_GUI_INVALID_CONTROL_ID;
    }
	return NULL;
}

// Find control and return pointer and panel id for it
//-----------------------------------------------------------------------------
CPUTResult CPUTGuiController::FindControl(CPUTControlID controlID, CPUTControl **ppControl, CPUTControlID *pPanelID)
{
    //if((NULL==ppControl) || (NULL==pPanelID))
        //return CPUT_ERROR_INVALID_PARAMETER;

    for(UINT i=0; i<mControlPanelIDList.size(); i++)
    {
        for(UINT j=0; j<mControlPanelIDList[i]->mControlList.size(); j++)
        {
            if(controlID == mControlPanelIDList[i]->mControlList[j]->GetControlID())
            {
                // found it!
                *pPanelID = mControlPanelIDList[i]->mpanelID;
                *ppControl = mControlPanelIDList[i]->mControlList[j];
                return CPUT_SUCCESS;
            }
        }
    }
    return CPUT_ERROR_NOT_FOUND;
}

// Delete all the controls in the list
//-----------------------------------------------------------------------------
void CPUTGuiController::DeleteAllControls()
{
    // set active panel to invalid
    mActiveControlPanelSlotID = CPUT_CONTROL_ID_INVALID;

    // walk list of panels deleting each list of controls
    int panelCount = (int) mControlPanelIDList.size();
    for(int i=0; i<panelCount; i++)
    {
        int controlListCount = (int)mControlPanelIDList[i]->mControlList.size();
        for(int j=0; j<controlListCount; j++)
        {
            SAFE_DELETE( mControlPanelIDList[i]->mControlList[j] );            
        }

        // erase this panel's control list
        mControlPanelIDList[i]->mControlList.clear();

        // delete the panel object
        SAFE_DELETE( mControlPanelIDList[i] );
        mControlPanelIDList[i] = NULL;
    }

    // clear the panel list
    mControlPanelIDList.clear();

    // trigger refresh
    if(mbAutoLayout)
    {
        Resize();
    }
}

// Flag the GUI system that a control has changed shape/size
// and that it needs to recalculate it's layout on the next render
//--------------------------------------------------------------------------------
void CPUTGuiController::Resize()
{
    mRecalculateLayout = true;
    mUberBufferDirty = true;
}

// Re-calculates all the positions of the controls based on their sizes
// to have a 'pretty' layout
//--------------------------------------------------------------------------------
void CPUTGuiController::RecalculateLayout()
{
    // if we have no valid panel, just return
    if(CPUT_CONTROL_ID_INVALID == mActiveControlPanelSlotID)
    {
        return;
    }

    // if we don't want the auto-layout feature, just return
    if(false == mbAutoLayout)
    {
        return;
    }

    // get window size
    CPUT_RECT windowRect;
    CPUTOSServices *pServices = CPUTOSServices::GetOSServices();
    pServices->GetClientDimensions(&windowRect.x, &windowRect.y, &windowRect.width, &windowRect.height);

    // Build columns of controls right to left
    int x,y;
    x=0; y=0;

    // walk list of controls, counting up their *heights*, until the
    // column is full.  While counting, keep track of the *widest*
    int width, height;
    const int GUI_WINDOW_PADDING = 5;

    int numberOfControls = (int) mControlPanelIDList[mActiveControlPanelSlotID]->mControlList.size();
    int indexStart=0;
    int indexEnd=0;
    int columnX = 0;
    int columnNumber = 1;
    while(indexEnd < numberOfControls)
    {
        int columnWidth=0;
        y=0;
        // figure out which controls belong in this column + column width
        while( indexEnd < numberOfControls )
        {
            if(mControlPanelIDList[mActiveControlPanelSlotID]->mControlList[indexEnd]->IsVisible() &&
                mControlPanelIDList[mActiveControlPanelSlotID]->mControlList[indexEnd]->IsAutoArranged())
            {
                mControlPanelIDList[mActiveControlPanelSlotID]->mControlList[indexEnd]->GetDimensions(width, height);
                if( y + height + GUI_WINDOW_PADDING < (windowRect.height-2*GUI_WINDOW_PADDING))
                {
                    y = y + height + GUI_WINDOW_PADDING;
                    if(columnWidth < width)
                    {
                        columnWidth = width;
                    }
                    indexEnd++;
                }
                else
                {
                    // if the window is now so small it won't fit a whole control, just
                    // draw one anyway and it'll just have to be clipped
                    if(indexEnd == indexStart)
                    {
                        columnWidth = width;
                        indexEnd++;
                    }
                    break;
                }
            }
            else
            {
                indexEnd++;
            }
        }
        

        // ok, now re-position each control with x at widest, and y at proper height
        y=GUI_WINDOW_PADDING;
        for(int i=indexStart; i<indexEnd; i++)
        {
            if(mControlPanelIDList[mActiveControlPanelSlotID]->mControlList[i]->IsVisible() &&
                mControlPanelIDList[mActiveControlPanelSlotID]->mControlList[i]->IsAutoArranged())
            {
                mControlPanelIDList[mActiveControlPanelSlotID]->mControlList[i]->GetDimensions(width, height);
                x = windowRect.width - columnX - columnWidth - (columnNumber*GUI_WINDOW_PADDING);
                mControlPanelIDList[mActiveControlPanelSlotID]->mControlList[i]->SetPosition(x,y);

                y = y + height + GUI_WINDOW_PADDING;
            }
        }
        indexStart = indexEnd;
        columnX+=columnWidth;
        columnNumber++;
    }
        
    mRecalculateLayout = false;
}

// Sets the object to call back for all newly created objects
// if ForceAll=true, then it walks the list of all the registered controls
// in all the panels and resets their callbacks
//-----------------------------------------------------------------------------
void CPUTGuiController::SetCallback(CPUTCallbackHandler *pHandler, bool ForceAll)
{
    if(true == ForceAll)
    {
        // walk list of ALL the controls and reset the callback pointer
        int panelCount = (int) mControlPanelIDList.size();
        for(int i=0; i<panelCount; i++)
        {
            int controlListCount = (int) mControlPanelIDList[i]->mControlList.size();
            for(int j=0; j<controlListCount; j++)
            {
                mControlPanelIDList[i]->mControlList[j]->SetControlCallback(pHandler);
            }
        }
    }
    else
    {
        // set the callback handler to be used on any NEW controls added
        mpHandler = pHandler;
    }
}
