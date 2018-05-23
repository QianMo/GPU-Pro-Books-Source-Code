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
#include "CPUTDropdown.h"

#include "CPUTText.h"

// uber-buffer
CPUT_SIZE CPUTDropdown::mpDropdownIdleImageSizeList[] = { {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0} };
CPUT_SIZE CPUTDropdown::mpDropdownDisabledSizeList[] = { {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0} };

// texture atlas information
float gDropdownAtlasWidth = 256.0f;
float gDropdownAtlasHeight = 64.0f;

// Texture atlas coordinates of the active-idle dropdown control
int gDropdownUVLocations_active[] = { 
    140,42, 143,45, // lt
    140,47, 143,48, // lm
    140,50, 143,53, // lb

    145,42, 146,45, // mt
    145,47, 146,48, // mm
    145,50, 146,53, // mb

    148,42, 151,45, // rt
    148,47, 151,48, // rm
    148,50, 151,53, // rb

    139,21, 159,41, // button
    163,21, 183,41, // button down
    230,21, 231,22  // highlight
};


// Texture atlas coordinates of the disabled dropdown control
int gDropdownUVLocations_disabled[] = { 
    155,42, 158,45, // lt
    155,47, 158,48, // lm
    155,50, 158,53, // lb

    160,42, 161,45, // mt
    160,47, 161,48, // mm
    160,50, 161,53, // mb

    163,42, 166,45, // rt
    163,47, 166,48, // rm
    163,50, 166,53, // rb

    186,21, 206,41, // button
    208,21, 228,41, // button down
    236,21, 237,22  // highlight
};
const int DropdownQuadCount = 12;
float3 mpDropdownUVCoords_active[DropdownQuadCount*2];
float3 mpDropdownUVCoords_disabled[DropdownQuadCount*2];

int CTrayLM = 1;
int CTrayLB = 2;
int CTrayMM = 4;
int CTrayMB = 5;
int CTrayRM = 7;
int CTrayRB = 8;
int CTrayHighlight = 11;



// Constructor
//-----------------------------------------------------------------------------
CPUTDropdown::CPUTDropdown(const cString ControlName, CPUTControlID id, CPUTFont *pFont):mVertexStride(0),
    mVertexOffset(0),
    mSelectedItemIndex(0),
    mConfirmedSelectedItemIndex(0),
    mbSizeDirty(false),
    mbMouseInside(false),
    mbStartedClickInside(false),
    mbStartedClickInsideTray(false),
    mRevertItem((UINT)-1),

    mpMirrorBufferActive(NULL),
    mpMirrorBufferDisabled(NULL),
    mpButtonText(NULL),
    mpFont(pFont)
{
    // initialize the state variables
    InitialStateSet();

    // save the control ID for callbacks
    mcontrolID = id;

    // set as enabled
    CPUTControl::SetEnable(true);

    // this is a copy of whatever item is selected
    mpSelectedItemCopy = new CPUTText(pFont);

        // clear the button selected area rect
    mButtonRect.x=0; mButtonRect.y=0; mButtonRect.width=0; mButtonRect.height=0;
    mTrayDimensions.x=0; mTrayDimensions.y=0; mTrayDimensions.width=0; mTrayDimensions.height=0;

    // set the string to display with the slider
    AddSelectionItem(ControlName, true);


    // Register any instance resources
    RegisterInstanceResources();
}

// Initial state of the control's member variables
//-----------------------------------------------------------------------------
void CPUTDropdown::InitialStateSet()
{
    // set the type
    mcontrolType = CPUT_DROPDOWN;
    mControlState = CPUT_CONTROL_ACTIVE;
    mControlGuiState = CPUT_DROPDOWN_GUI_MOUSE_NEUTRAL;

    // save the control ID for callbacks
    mcontrolID = 0;

    // default location
    mControlDimensions.x=0;
    mControlDimensions.y=0;
    mControlDimensions.height=0;
    mControlDimensions.width=0;
}

// Destructor
//------------------------------------------------------------------------------
CPUTDropdown::~CPUTDropdown()
{
    UnRegisterInstanceResources();
}

// Return xy position of control
//-----------------------------------------------------------------------------
void CPUTDropdown::GetPosition(int &x, int &y)
{
    x = mControlDimensions.x;
    y = mControlDimensions.y;
}

// Returns the number of quads needed to draw this control
//--------------------------------------------------------------------------------
unsigned int CPUTDropdown::GetOutputVertexCount()
{
    // A dropdown is always made of:
    //   - 9 quads for the display area
    //   - 1 quad for the tray button
    //   - 6 quads for the tray itself
    //   - 1 quad for the highlighted area 

    // Quads 2,4,5,6, and 8 'stretch' in height and/or width to fit the 
    // static text/content inside the button
    //
    //   ---+------------------+---
    //  | 1 |     4            | 7 |              
    //  |---+------------------+---|
    //  |   |           +----+ |   | 
    //  | 2 |     5     | 10 | | 8 |              
    //  |   |           +----+ |   | 
    //  |---+------------------+---|
    //  | 3 |     6            | 9 |
    //   ---+------------------+---
    //    |   |              |   | 
    //    | 11|     12       | 13|              
    //    |   |              |   | 
    //    |---+--------------+---|
    //    | 14|     15       | 16|
    //     ---+--------------+---
    //
    // calculation: (3 verts/triangle * 2 triangles/quad) * (9 + 1 + 6 + 1 quads)

    return (3*2) * (9+1+6+1); 
}





// Load and register all instance variables
//-----------------------------------------------------------------------------
CPUTResult CPUTDropdown::RegisterInstanceResources()
{
    return CPUT_SUCCESS;
}

// Release all instance resources (vertex buffer quads/etc)
//-----------------------------------------------------------------------------
CPUTResult CPUTDropdown::UnRegisterInstanceResources()
{
    // delete the list of selection items (CPUTTexts)
    for(UINT i=0; i<mpListOfSelectableItems.size(); i++)
    {
        // physically delete the text object
        CPUTText *pItem = mpListOfSelectableItems[i];
        SAFE_DELETE(pItem);        
    }

    // clear the list
    mpListOfSelectableItems.clear();

    // delete vertex mirror buffers
    SAFE_DELETE_ARRAY(mpMirrorBufferActive);
    SAFE_DELETE_ARRAY(mpMirrorBufferDisabled);
    
    // delete the extra static item used for display readout
    SAFE_DELETE(mpSelectedItemCopy);
    
    return CPUT_SUCCESS;
}



// Add an item to the selection list
//-----------------------------------------------------------------------------
CPUTResult CPUTDropdown::AddSelectionItem(const cString Item, bool bIsSelected)
{
    // Don't allow adding 'empty' elements
    // causes a spacing issue
    if(0==Item.size())
    {
        return CPUT_ERROR_INVALID_PARAMETER;
    }

    // create a static item to hold the text
    CPUTText *pNewItem = new CPUTText(mpFont);
    pNewItem->SetText(Item, 0.35f); // set the text slightly higher than the 0.5 normal mode

    mpListOfSelectableItems.push_back(pNewItem);

    // Should this be the currently selected item?
    if(bIsSelected)
    {
        mSelectedItemIndex = (int) mpListOfSelectableItems.size()-1;    // set the highlighted element to this item
        mConfirmedSelectedItemIndex = mSelectedItemIndex;               // set the hard-selected item to this as well
    }
    // was this the first item added to an empty list?
    if( 1==mpListOfSelectableItems.size() )
    {
        mSelectedItemIndex=0;
        mConfirmedSelectedItemIndex=0;
    }

    // mark that a resize of dropdown selection box is needed
    mbSizeDirty = true;

    Recalculate();

    // position or size may move - force a recalculation of this control's location
    // if it is managed by the auto-arrange function
    if(this->IsAutoArranged())
    {
        mControlNeedsArrangmentResizing = true;
    }

    return CPUT_SUCCESS;
}

// Return the number of items in the dropdown list
//-----------------------------------------------------------------------------
void CPUTDropdown::NumberOfSelectableItems(UINT &count)
{
    count = (UINT) mpListOfSelectableItems.size();
}

// Return the index of the currently selected item
//-----------------------------------------------------------------------------
void CPUTDropdown::GetSelectedItem(UINT &index)
{
    //index = mSelectedItemIndex+1;
    index = mConfirmedSelectedItemIndex;
}

// Fill user's buffer with string of the currently selected item
//-----------------------------------------------------------------------------
void CPUTDropdown::GetSelectedItem(cString &Item)
{
    if(-1!=mSelectedItemIndex)
    {
        //mpListOfSelectableItems[mSelectedItemIndex]->GetString(Item);
        mpListOfSelectableItems[mConfirmedSelectedItemIndex]->GetString(Item);
    }
}

// Sets which item in the dropdown is currently selected
//-----------------------------------------------------------------------------
void CPUTDropdown::SetSelectedItem(const UINT index)
{
    if( (index>0) && (index <= mpListOfSelectableItems.size() ))
    {
        mSelectedItemIndex = (index-1);
        mConfirmedSelectedItemIndex = mSelectedItemIndex; // set the hard-selected item to this as well
    }

    // Recalculate and mark control for redraw
    Recalculate();
}

// Sets which item in the dropdown is currently highlighted
//-----------------------------------------------------------------------------
void CPUTDropdown::SetHighlightedItem(const UINT index)
{
    if( (index>0) && (index <= mpListOfSelectableItems.size() ))
    {
        mSelectedItemIndex = (index-1);
    }

    // Recalculate and mark control for redraw
    Recalculate(); 
}

// Delete an item at a specific index
//-----------------------------------------------------------------------------
void CPUTDropdown::DeleteSelectionItem(const UINT index)
{
    if((0==index) || (index>mpListOfSelectableItems.size()))
        return;

    // list is zero based, not 1 based
    UINT itemIndex = index-1;

    // physically delete the text object
    CPUTText *pItem = mpListOfSelectableItems[itemIndex];
    SAFE_DELETE(pItem);    

    // remove the item from the list
    mpListOfSelectableItems.erase(mpListOfSelectableItems.begin()+itemIndex);

    // if we're deleting the selected item or any above it
    // move the current selection down by oneq
    if(mSelectedItemIndex > itemIndex)
    {
        mSelectedItemIndex--;
        mConfirmedSelectedItemIndex = mSelectedItemIndex;
    }
    if(mSelectedItemIndex >= mpListOfSelectableItems.size())
    {
        mSelectedItemIndex = (UINT) mpListOfSelectableItems.size()-1;
        mConfirmedSelectedItemIndex = mSelectedItemIndex;
    }

    // resize the dropdown selection box
    mbSizeDirty = true;

    Recalculate();

    // position or size may move - force a recalculation of this control's location
    // if it is managed by the auto-arrange function
    if(this->IsAutoArranged())
    {
        mControlNeedsArrangmentResizing = true;
    }
}

// Delete an item with string
// Only deletes the first occurance of the string, you'd need to call multiple
// times to delete each occurance
//-----------------------------------------------------------------------------
void CPUTDropdown::DeleteSelectionItem(const cString string)
{
    cString itemText;

    for(UINT i=0; i<mpListOfSelectableItems.size(); i++)
    {
        mpListOfSelectableItems[i]->GetString(itemText);
        if(0==itemText.compare(string))
        {
            DeleteSelectionItem(i+1);
            break;
        }
    }

    Recalculate();
}


// Handle mouse events
//-----------------------------------------------------------------------------
CPUTEventHandledCode CPUTDropdown::HandleMouseEvent(int x, int y, int wheel, CPUTMouseState state)
{
    UNREFERENCED_PARAMETER(wheel);
    CPUTEventHandledCode handledCode = CPUT_EVENT_UNHANDLED;

    // just return if the control is disabled or invisible
    if((CPUT_CONTROL_INACTIVE == mControlState) || (false == mControlVisible) )
    {
        mbMouseInside = false;
        return handledCode;
    }

    // tray is down, commence to selecting
    if(CPUT_DROPDOWN_GUI_MOUSE_PRESSED == mControlGuiState) 
    {
        if(ContainsPointTrayArea(x, y))
        {
            // tray is open
            // float the selection up/down based on mouse position, making
            // whichever item the mouse is over the selected item
            CPUT_RECT inner, outer;
            CalculateTrayRect(inner, outer);
            if( (x>outer.x) && (x<outer.x+outer.width) &&
                (y>inner.y) && (y<inner.y+inner.height))
            {
                // figure out which one was selected
                int itemWidth, itemHeight;
                CalculateMaxItemSize(itemWidth, itemHeight);
                int itemSelect = (int)( (y - inner.y)  / (float)(itemHeight+2*CPUT_DROPDOWN_TEXT_PADDING));
                SetHighlightedItem(itemSelect+1);
            }

            if( CPUT_MOUSE_LEFT_DOWN == state )
            {
                mbStartedClickInsideTray = true;
            }

            // released the left-down button, but there was a click started here
            if( !(CPUT_MOUSE_LEFT_DOWN == state) && (true == mbStartedClickInsideTray) )
            {
                // mark click start as no longer active
                mbStartedClickInsideTray = false;

                // close the tray
                //mControlState = CPUT_CONTROL_ACTIVE;
                mControlGuiState = CPUT_DROPDOWN_GUI_MOUSE_NEUTRAL;
                Recalculate();

                // set the hard-selected item
                mConfirmedSelectedItemIndex = mSelectedItemIndex;

                // fire callback
                mpCallbackHandler->HandleCallbackEvent(1, mcontrolID, (CPUTControl*) this);
                
                // remember this item as the last selected item
                GetSelectedItem(mRevertItem);
            }

            // anything in the tray counts as handling the mouse event
            return CPUT_EVENT_HANDLED;
        }
    }

    // handle just the readout area
    if(ContainsPointReadoutArea(x,y))
    {
        // click started outside this control - ignore it
        if( (false == mbMouseInside ) && (CPUT_MOUSE_LEFT_DOWN == state) )
        {
            return handledCode;
        }

        // clicked and dragging around inside the control itself - ignore
        if( (true == mbStartedClickInside ) && (CPUT_MOUSE_LEFT_DOWN == state) )
        {
            return CPUT_EVENT_HANDLED;
        }

        mbMouseInside = true;
        if( !(CPUT_MOUSE_LEFT_DOWN == state))
        {
            mbStartedClickInside = false;
            mbStartedClickInsideTray = false;
        }

        if( CPUT_MOUSE_LEFT_DOWN == state )
        {
            if(CPUT_DROPDOWN_GUI_MOUSE_NEUTRAL == mControlGuiState) 
            {
                mbStartedClickInside = true;
                mbStartedClickInsideTray = true;

                // get the item to revert to
                GetSelectedItem(mRevertItem);

                // toggle tray state
                mControlGuiState = CPUT_DROPDOWN_GUI_MOUSE_PRESSED;
                Recalculate();

            }
            else
            {
                mbStartedClickInside = true;
                mbStartedClickInsideTray = true;
                //mControlState = CPUT_CONTROL_ACTIVE;
                mControlGuiState = CPUT_DROPDOWN_GUI_MOUSE_NEUTRAL;
                SetSelectedItem(mRevertItem);
                mRevertItem=(UINT)-1;
                Recalculate();
            }

            return CPUT_EVENT_HANDLED;
        }
    }
    else if(CPUT_MOUSE_LEFT_DOWN == state)
    {
        // clicked outside the control
        // if tray is down, restore previous selection and disable
        // display of the tray
        if(CPUT_DROPDOWN_GUI_MOUSE_NEUTRAL != mControlGuiState)
        {
            mControlGuiState = CPUT_DROPDOWN_GUI_MOUSE_NEUTRAL;
            SetSelectedItem(mRevertItem+1);
            mRevertItem=(UINT)-1;
            Recalculate();
        }
        
        mbStartedClickInside = false;
        mbMouseInside = false;
        return CPUT_EVENT_UNHANDLED;
    }
    return handledCode;
}

// Register static textures/etc for this control
//-----------------------------------------------------------------------------
CPUTResult CPUTDropdown::RegisterStaticResources()
{
    // calculate the UV coordinates of each of the 12 images that
    // make up a button.  Do this for the active, pressed, and disabled states.
    for(int ii=0; ii<DropdownQuadCount*2; ii++)
    {
        mpDropdownUVCoords_active[ii].x = gDropdownUVLocations_active[2*ii]/gDropdownAtlasWidth;
        mpDropdownUVCoords_active[ii].y = gDropdownUVLocations_active[2*ii+1]/gDropdownAtlasHeight;

        mpDropdownUVCoords_disabled[ii].x = gDropdownUVLocations_disabled[2*ii]/gDropdownAtlasWidth;
        mpDropdownUVCoords_disabled[ii].y = gDropdownUVLocations_disabled[2*ii+1]/gDropdownAtlasHeight;
    }

    // calculate the width/height in pixels of each of the 9 image slices
    // that makes up the button images
    int QuadIndex=0;
    for(int ii=0; ii<DropdownQuadCount*4; ii+=4)
    {
        mpDropdownIdleImageSizeList[QuadIndex].width = gDropdownUVLocations_active[ii+2] - gDropdownUVLocations_active[ii+0];
        mpDropdownIdleImageSizeList[QuadIndex].height = gDropdownUVLocations_active[ii+3] - gDropdownUVLocations_active[ii+1];

        mpDropdownDisabledSizeList[QuadIndex].width = gDropdownUVLocations_disabled[ii+2] - gDropdownUVLocations_disabled[ii+0];
        mpDropdownDisabledSizeList[QuadIndex].height = gDropdownUVLocations_disabled[ii+3] - gDropdownUVLocations_disabled[ii+1];
        QuadIndex++;
    }


    return CPUT_SUCCESS;
}

// Release all statically register resources
//-----------------------------------------------------------------------------
CPUTResult CPUTDropdown::UnRegisterStaticResources()
{
    // mpDropdownUVCoords_* and mpDropdown*ImageSizeList are both statically allocated
    // nothing to dealloc anymore
    return CPUT_SUCCESS;
}

//
//-----------------------------------------------------------------------------
void CPUTDropdown::CalculateButtonRect(CPUT_RECT &button)
{
    CPUT_RECT inner, outer;
    CalculateReadoutRect(inner, outer);

    button.x =  (inner.x + inner.width) - mpDropdownIdleImageSizeList[CButtonUp].width - CPUT_DROPDOWN_TEXT_PADDING;
    button.y =  mControlDimensions.y + mpDropdownIdleImageSizeList[CLeftTop].height;

    button.width = mpDropdownIdleImageSizeList[CButtonUp].width;
    button.height = mpDropdownIdleImageSizeList[CButtonUp].height;

    mButtonRect.x = button.x;
    mButtonRect.y = button.y;
    mButtonRect.width = button.width;
    mButtonRect.height = button.height;
}

//
//-----------------------------------------------------------------------------
void CPUTDropdown::CalculateReadoutRect(CPUT_RECT &inner, CPUT_RECT &outer)
{
    bool ButtonIsBiggerThanText = false;

    // calculate outer dimensions
    outer.x = mControlDimensions.x;
    outer.y = mControlDimensions.y;
    int maxItemSizeWidth, maxItemSizeHeight;

    CalculateMaxItemSize(maxItemSizeWidth, maxItemSizeHeight);
    if( (maxItemSizeHeight+2*CPUT_DROPDOWN_TEXT_PADDING) < mpDropdownIdleImageSizeList[CButtonUp].height)
    {
        ButtonIsBiggerThanText = true;
    }

    outer.width = mpDropdownIdleImageSizeList[CLeftTop].width + CPUT_DROPDOWN_TEXT_PADDING + maxItemSizeWidth + CPUT_DROPDOWN_TEXT_PADDING + mpDropdownIdleImageSizeList[CButtonUp].width + mpDropdownIdleImageSizeList[CRightTop].width;
    if(ButtonIsBiggerThanText)
    {
        outer.height = mpDropdownIdleImageSizeList[CLeftTop].height + mpDropdownIdleImageSizeList[CButtonUp].width + mpDropdownIdleImageSizeList[CLeftBot].height;
    }
    else
    {
        outer.height = mpDropdownIdleImageSizeList[CLeftTop].height + CPUT_DROPDOWN_TEXT_PADDING + maxItemSizeHeight + CPUT_DROPDOWN_TEXT_PADDING + mpDropdownIdleImageSizeList[CLeftBot].height;
    }

    // calculate the inner dimensions
    inner.x = mControlDimensions.x + mpDropdownIdleImageSizeList[CLeftTop].width + CPUT_DROPDOWN_TEXT_PADDING;
    inner.y = mControlDimensions.y + mpDropdownIdleImageSizeList[CLeftTop].height;

    if(ButtonIsBiggerThanText)
    {
        inner.width = CPUT_DROPDOWN_TEXT_PADDING + maxItemSizeWidth + CPUT_DROPDOWN_TEXT_PADDING + mpDropdownIdleImageSizeList[CButtonUp].width;
        inner.height = mpDropdownIdleImageSizeList[CButtonUp].height;
    }
    else
    {
        inner.width = CPUT_DROPDOWN_TEXT_PADDING + maxItemSizeWidth + CPUT_DROPDOWN_TEXT_PADDING + mpDropdownIdleImageSizeList[CButtonUp].width;
        inner.height = maxItemSizeHeight + 2*CPUT_DROPDOWN_TEXT_PADDING;
    }
}

//
//-----------------------------------------------------------------------------
void CPUTDropdown::CalculateReadoutTextPosition(int &x, int &y)
{
    CPUT_RECT inner, outer;
    CalculateReadoutRect(inner, outer);

    CPUT_RECT textDimensions;
    if(0==mpListOfSelectableItems.size())
    {
        textDimensions.height=1;
    }
    else
    {
        mpListOfSelectableItems[mSelectedItemIndex]->GetDimensions(textDimensions.width, textDimensions.height);
    }

    x = inner.x;
    y = (int) (inner.y + inner.height/2.0f - textDimensions.height/2.0f + CPUT_DROPDOWN_TEXT_PADDING/2.0f);

}

//
//-----------------------------------------------------------------------------
void CPUTDropdown::CalculateMaxItemSize(int &width, int &height)
{
    width=0; height=0;

    if(mpListOfSelectableItems.size())
    {
        for(UINT i=0; i<mpListOfSelectableItems.size(); i++)
        {
            CPUT_RECT rect;
            mpListOfSelectableItems[i]->GetDimensions(rect.width, rect.height);
            if(width < rect.width)
            {
                width = rect.width;
            }
            if(height < rect.height)
            {
                height = rect.height;
            }
        }
    }
    else
    {
        // give dropdown a nice 'minimum' size if there is nothing
        mpSelectedItemCopy->SetText(_L("                    "));
        mpSelectedItemCopy->GetDimensions(width, height);
    }
}

//
//-----------------------------------------------------------------------------
void CPUTDropdown::CalculateTrayRect(CPUT_RECT &inner, CPUT_RECT &outer)
{
    int maxItemWidth, maxItemHeight;
    CalculateMaxItemSize(maxItemWidth, maxItemHeight);

    CPUT_RECT innerReadout, outerReadout;
    CalculateReadoutRect(innerReadout, outerReadout);

    // locations
    outer.x = outerReadout.x + 5;
    outer.y = outerReadout.y + outerReadout.height;
    inner.x = outer.x + mpDropdownIdleImageSizeList[CLeftMid].width;
    inner.y = outer.y + CPUT_DROPDOWN_TEXT_PADDING;

    // dimension
    inner.width = maxItemWidth + 2*CPUT_DROPDOWN_TEXT_PADDING;
    inner.height = (int)((mpListOfSelectableItems.size())*(maxItemHeight+2*CPUT_DROPDOWN_TEXT_PADDING));

    outer.width = inner.width + mpDropdownIdleImageSizeList[CLeftMid].width + mpDropdownIdleImageSizeList[CRightMid].width;
    outer.height = inner.height + mpDropdownIdleImageSizeList[CLeftBot].height;
}


// Enable/disable the control
//--------------------------------------------------------------------------------
void CPUTDropdown::SetEnable(bool in_bEnabled)
{
    if(in_bEnabled)
    {
        mControlState = CPUT_CONTROL_ACTIVE;
    }
    else
    {
        mControlState = CPUT_CONTROL_INACTIVE;
    }

    // set the control's text to match
    mpSelectedItemCopy->SetEnable(in_bEnabled);

    // recalculate control's quads 
    Recalculate();
}

// Get enabled/disabled state of this control
//-----------------------------------------------------------------------------
bool CPUTDropdown::IsEnabled()
{
    if(CPUT_CONTROL_INACTIVE == mControlState)
    {
        return false;
    }

    return true;
}

//
//-----------------------------------------------------------------------------
bool CPUTDropdown::ContainsPoint(int x, int y)
{
    // clicking the readout box
    CPUT_RECT inner, outer;
    CalculateReadoutRect(inner, outer);
    if( (x>outer.x) &&
        (x<outer.x+outer.width) &&
        (y>outer.y) &&
        (y<outer.y+outer.height) )
        return true;

    // this is the dimensions of just the top selection
    // tray = closed
    if(CPUT_CONTROL_ACTIVE == mControlState)
    {
        if(CPUT_DROPDOWN_GUI_MOUSE_NEUTRAL == mControlGuiState)
        {
            // button?
            if( (x< mButtonRect.x) ||
                (x> mButtonRect.x+mButtonRect.width) ||
                (y< mButtonRect.y) ||
                (y> mButtonRect.y+mButtonRect.height) )
            {
                return false;
            }
            return true;
        }

        // this is dimensions of top selection + tray
        // tray = open
        if(CPUT_DROPDOWN_GUI_MOUSE_PRESSED == mControlGuiState ) //CPUT_CONTROL_PRESSED == mControlState)
        {
            // in button area?
            if( (x> mButtonRect.x) &&
                (x< mButtonRect.x+mButtonRect.width) &&
                (y> mButtonRect.y) &&
                (y< mButtonRect.y+mButtonRect.height) )
            {
                return true;
            }

            // in the tray area?
            CPUT_RECT inner, outer;
            CalculateTrayRect(inner, outer);

            if( (x>outer.x) &&
                (x<outer.x+outer.width) &&
                (y>outer.y) &&
                (y<outer.y+outer.height) )
            {
                return true;
            }
        }
    }
    return false;
}

//
//-----------------------------------------------------------------------------
bool CPUTDropdown::ContainsPointReadoutArea(int x, int y)
{

    CPUT_RECT inner, outer;
    CalculateReadoutRect(inner, outer);

    // in the readout box area?
    if( (x>outer.x) &&
        (x<outer.x+outer.width) &&
        (y>outer.y) &&
        (y<outer.y+outer.height) )
    {
        return true;
    }

    // button area?
    if( (x> mButtonRect.x) &&
        (x< mButtonRect.x+mButtonRect.width) &&
        (y> mButtonRect.y) &&
        (y< mButtonRect.y+mButtonRect.height) )
    {
        return true;
    }

    return false;
}

//
//-----------------------------------------------------------------------------
bool CPUTDropdown::ContainsPointTrayArea(int x, int y)
{
    // this is dimensions of top selection + tray
    // tray = open
    if(CPUT_CONTROL_ACTIVE == mControlState)
    {
        if(CPUT_DROPDOWN_GUI_MOUSE_PRESSED == mControlGuiState)
        {
            // in the tray area?
            CPUT_RECT inner, outer;
            CalculateTrayRect(inner, outer);

            if( (x>outer.x) &&
                (x<outer.x+outer.width) &&
                (y>outer.y-1) &&
                (y<outer.y+outer.height) )
            {
                return true;
            }
        }
    }
    return false;
}

//
//-----------------------------------------------------------------------------
void CPUTDropdown::SetPosition(int x, int y)
{
    mControlDimensions.x = x;
    mControlDimensions.y = y;

    int a,b;
    CalculateReadoutTextPosition(a,b);
    if(-1 != mSelectedItemIndex)
    {
        mpListOfSelectableItems[mSelectedItemIndex]->SetPosition(a,b);
    }

    Recalculate();
}

//
//-----------------------------------------------------------------------------
void CPUTDropdown::GetDimensions(int &width, int &height)
{
    CPUT_RECT inner, outer;
    CalculateReadoutRect(inner, outer);
    width = outer.width;
    height = outer.height;
}

// Recalculate all the quads used to draw this control
//--------------------------------------------------------------------------------
void CPUTDropdown::Recalculate()
{
    mpDropdownIdleSizeList;

    // calculate height/width of dropdown's interior based on string or button size
    CPUT_RECT inner, outer;
    CalculateReadoutRect(inner, outer);

    // calculate the pieces we'll need to rebuild
    int middleWidth = inner.width;
    int middleHeight = inner.height;

    // TODO: this should be calculated like 
    const int mSmallestTopSizeIdle = mpDropdownIdleImageSizeList[0].height;
    const int mSmallestLeftSizeIdle =mpDropdownIdleImageSizeList[0].width;


    // clear and allocate the buffers
    const int VertexCount = 18*3*2;
    SAFE_DELETE_ARRAY(mpMirrorBufferActive);
    mpMirrorBufferActive = new CPUTGUIVertex[VertexCount];
        
    SAFE_DELETE_ARRAY(mpMirrorBufferDisabled);
    mpMirrorBufferDisabled = new CPUTGUIVertex[VertexCount];

    //
    // delete the old button quads for the middle sections
    //result = UnRegisterResizableInstanceQuads();

    // Calculate the selected item area images
    // Idle button quads
    // left
    AddQuadIntoMirrorBuffer(mpMirrorBufferActive, 0*6, (float)mControlDimensions.x, (float)mControlDimensions.y, (float) mpDropdownIdleImageSizeList[0].width, (float) mpDropdownIdleImageSizeList[0].height, mpDropdownUVCoords_active[2*0], mpDropdownUVCoords_active[2*0+1] );
    AddQuadIntoMirrorBuffer(mpMirrorBufferActive, 1*6, (float)mControlDimensions.x, (float)mControlDimensions.y+mSmallestTopSizeIdle, (float) mpDropdownIdleImageSizeList[1].width, (float) middleHeight, mpDropdownUVCoords_active[2*1], mpDropdownUVCoords_active[2*1+1] );
    mpDropdownIdleSizeList[1].height = middleHeight;
    AddQuadIntoMirrorBuffer(mpMirrorBufferActive, 2*6, (float)mControlDimensions.x, (float)mControlDimensions.y+mSmallestTopSizeIdle+middleHeight, (float) mpDropdownIdleImageSizeList[2].width, (float) mpDropdownIdleImageSizeList[2].height, mpDropdownUVCoords_active[2*2], mpDropdownUVCoords_active[2*2+1] );

    // middle
    AddQuadIntoMirrorBuffer(mpMirrorBufferActive, 3*6, (float)mControlDimensions.x+mSmallestLeftSizeIdle, (float)mControlDimensions.y, (float) middleWidth, (float) mpDropdownIdleImageSizeList[3].height, mpDropdownUVCoords_active[2*3], mpDropdownUVCoords_active[2*3+1] );
    mpDropdownIdleSizeList[3].width = middleWidth;
    AddQuadIntoMirrorBuffer(mpMirrorBufferActive, 4*6, (float)mControlDimensions.x+mSmallestLeftSizeIdle, (float)mControlDimensions.y+mSmallestTopSizeIdle, (float) middleWidth, (float) middleHeight, mpDropdownUVCoords_active[2*4], mpDropdownUVCoords_active[2*4+1] );
    mpDropdownIdleSizeList[4].width = middleWidth;
    mpDropdownIdleSizeList[4].height = middleHeight;
    AddQuadIntoMirrorBuffer(mpMirrorBufferActive, 5*6, (float)mControlDimensions.x+mSmallestLeftSizeIdle, (float)mControlDimensions.y+mSmallestTopSizeIdle+middleHeight, (float) middleWidth, (float) mpDropdownIdleImageSizeList[5].height, mpDropdownUVCoords_active[2*5], mpDropdownUVCoords_active[2*5+1] );
    mpDropdownIdleSizeList[5].width = middleWidth;

    // right
    AddQuadIntoMirrorBuffer(mpMirrorBufferActive, 6*6, (float)mControlDimensions.x+mSmallestLeftSizeIdle+middleWidth, (float)mControlDimensions.y, (float) mpDropdownIdleImageSizeList[6].width, (float)mpDropdownIdleImageSizeList[6].height, mpDropdownUVCoords_active[2*6], mpDropdownUVCoords_active[2*6+1] );
    AddQuadIntoMirrorBuffer(mpMirrorBufferActive, 7*6, (float)mControlDimensions.x+mSmallestLeftSizeIdle+middleWidth, (float)mControlDimensions.y+mSmallestTopSizeIdle, (float) mpDropdownIdleImageSizeList[7].width, (float)middleHeight, mpDropdownUVCoords_active[2*7], mpDropdownUVCoords_active[2*7+1] );
    mpDropdownIdleSizeList[7].height = middleHeight;
    AddQuadIntoMirrorBuffer(mpMirrorBufferActive, 8*6, (float)mControlDimensions.x+mSmallestLeftSizeIdle+middleWidth, (float)mControlDimensions.y+mSmallestTopSizeIdle+middleHeight, (float) mpDropdownIdleImageSizeList[8].width, (float)mpDropdownIdleImageSizeList[8].height, mpDropdownUVCoords_active[2*8], mpDropdownUVCoords_active[2*8+1] );

    // 2 buttons (up and pressed) at same location
    CPUT_RECT rect;
    CalculateButtonRect(rect);
    AddQuadIntoMirrorBuffer(mpMirrorBufferActive,  9*6, (float)rect.x, (float)rect.y, (float) mpDropdownIdleImageSizeList[ 9].width, (float)mpDropdownIdleImageSizeList[ 9].height, mpDropdownUVCoords_active[2*9], mpDropdownUVCoords_active[2*9+1] );
    AddQuadIntoMirrorBuffer(mpMirrorBufferActive, 10*6, (float)rect.x, (float)rect.y, (float) mpDropdownIdleImageSizeList[10].width, (float)mpDropdownIdleImageSizeList[10].height, mpDropdownUVCoords_active[2*10], mpDropdownUVCoords_active[2*10+1] );

    // if the dropdown tray is active, draw it
    int index=11;
    if(CPUT_CONTROL_ACTIVE == mControlState)
    {
        if((CPUT_DROPDOWN_GUI_MOUSE_PRESSED == mControlGuiState) && (0 != mpListOfSelectableItems.size()))
        {
            CalculateTrayRect(inner, outer);
            mTrayDimensions.width = inner.width;
            mTrayDimensions.height = inner.height;

            float tx, ty;
            tx = (float)outer.x;
            ty = (float)outer.y;

            CalculateMaxItemSize(mTrayDimensions.width, mTrayDimensions.height);
            mTrayDimensions.width += 2*CPUT_DROPDOWN_TEXT_PADDING;
            mTrayDimensions.height = inner.height;
            

            // lm
            AddQuadIntoMirrorBuffer(mpMirrorBufferActive,  index*6, (float)tx, (float)ty, (float) mpDropdownIdleImageSizeList[CTrayLM].width, (float)mTrayDimensions.height, mpDropdownUVCoords_active[2*CTrayLM], mpDropdownUVCoords_active[2*CTrayLM+1] );
            index++;

            // lb
            ty+=mTrayDimensions.height; // inner.height
            AddQuadIntoMirrorBuffer(mpMirrorBufferActive,  index*6, (float)tx, (float)ty, (float) mpDropdownIdleImageSizeList[CTrayLB].width, (float)mpDropdownIdleImageSizeList[CTrayLB].height, mpDropdownUVCoords_active[2*CTrayLB], mpDropdownUVCoords_active[2*CTrayLB+1] );
            index++;

            // mm
            tx+=mpDropdownIdleImageSizeList[CLeftMid].width;
            ty = (float)outer.y;
            AddQuadIntoMirrorBuffer(mpMirrorBufferActive,  index*6, (float)tx, (float)ty, (float) mTrayDimensions.width, (float)mTrayDimensions.height, mpDropdownUVCoords_active[2*CTrayMM], mpDropdownUVCoords_active[2*CTrayMM+1] );
            index++;

            // mb
            ty+=mTrayDimensions.height;
            AddQuadIntoMirrorBuffer(mpMirrorBufferActive,  index*6, (float)tx, (float)ty, (float) mTrayDimensions.width, (float)mpDropdownIdleImageSizeList[CTrayMB].height, mpDropdownUVCoords_active[2*CTrayMB], mpDropdownUVCoords_active[2*CTrayMB+1] );
            index++;


            // rm
            tx+=mTrayDimensions.width;
            ty = (float)outer.y;
            AddQuadIntoMirrorBuffer(mpMirrorBufferActive,  index*6, (float)tx, (float)ty, (float) mpDropdownIdleImageSizeList[CTrayRM].width, (float)mTrayDimensions.height, mpDropdownUVCoords_active[2*CTrayRM], mpDropdownUVCoords_active[2*CTrayRM+1] );
            index++;

            // rb
            ty += mTrayDimensions.height;
            AddQuadIntoMirrorBuffer(mpMirrorBufferActive,  index*6, (float)tx, (float)ty, (float) mpDropdownIdleImageSizeList[CTrayRB].width, (float)mpDropdownIdleImageSizeList[CTrayRB].height, mpDropdownUVCoords_active[2*CTrayRB], mpDropdownUVCoords_active[2*CTrayRB+1] );
            index++;


            // tray highlight
            int MaxSizeItemWidth, MaxSizeItemHeight;
            CalculateMaxItemSize(MaxSizeItemWidth, MaxSizeItemHeight);

            int sx = (inner.x + mpDropdownIdleImageSizeList[CLeftMid].width);
            int sy = (inner.y + CPUT_DROPDOWN_TEXT_PADDING);

            for(UINT i=0;i<mpListOfSelectableItems.size();i++)
            {
                if( (mSelectedItemIndex) == i )
                {
                    //m = XMMatrixTranslation((float)(sx - CPUT_DROPDOWN_TEXT_PADDING), (float)(sy - CPUT_DROPDOWN_TEXT_PADDING), 0);
                    AddQuadIntoMirrorBuffer(mpMirrorBufferActive,  index*6, (float)(sx - CPUT_DROPDOWN_TEXT_PADDING), (float)(sy - CPUT_DROPDOWN_TEXT_PADDING), (float) MaxSizeItemWidth + 2*CPUT_DROPDOWN_TEXT_PADDING, (float)MaxSizeItemHeight + CPUT_DROPDOWN_TEXT_PADDING, mpDropdownUVCoords_active[2*CTrayHighlight], mpDropdownUVCoords_active[2*CTrayHighlight+1] );
                    index++;
                }

                CPUT_RECT rect;
                mpListOfSelectableItems[i]->GetDimensions(rect.width, rect.height);
                sy+=rect.height + 2*CPUT_DROPDOWN_TEXT_PADDING;
            }
        }
    }

    // calculate the disabled item list
        // left
    AddQuadIntoMirrorBuffer(mpMirrorBufferDisabled, 0*6, (float)mControlDimensions.x, (float)mControlDimensions.y, (float) mpDropdownIdleImageSizeList[0].width, (float) mpDropdownIdleImageSizeList[0].height, mpDropdownUVCoords_disabled[2*0], mpDropdownUVCoords_disabled[2*0+1] );
    AddQuadIntoMirrorBuffer(mpMirrorBufferDisabled, 1*6, (float)mControlDimensions.x, (float)mControlDimensions.y+mSmallestTopSizeIdle, (float) mpDropdownIdleImageSizeList[1].width, (float) middleHeight, mpDropdownUVCoords_disabled[2*1], mpDropdownUVCoords_disabled[2*1+1] );
    mpDropdownIdleSizeList[1].height = middleHeight;
    AddQuadIntoMirrorBuffer(mpMirrorBufferDisabled, 2*6, (float)mControlDimensions.x, (float)mControlDimensions.y+mSmallestTopSizeIdle+middleHeight, (float) mpDropdownIdleImageSizeList[2].width, (float) mpDropdownIdleImageSizeList[2].height, mpDropdownUVCoords_disabled[2*2], mpDropdownUVCoords_disabled[2*2+1] );

    // middle
    AddQuadIntoMirrorBuffer(mpMirrorBufferDisabled, 3*6, (float)mControlDimensions.x+mSmallestLeftSizeIdle, (float)mControlDimensions.y, (float) middleWidth, (float) mpDropdownIdleImageSizeList[3].height, mpDropdownUVCoords_disabled[2*3], mpDropdownUVCoords_disabled[2*3+1] );
    mpDropdownIdleSizeList[3].width = middleWidth;
    AddQuadIntoMirrorBuffer(mpMirrorBufferDisabled, 4*6, (float)mControlDimensions.x+mSmallestLeftSizeIdle, (float)mControlDimensions.y+mSmallestTopSizeIdle, (float) middleWidth, (float) middleHeight, mpDropdownUVCoords_disabled[2*4], mpDropdownUVCoords_disabled[2*4+1] );
    mpDropdownIdleSizeList[4].width = middleWidth;
    mpDropdownIdleSizeList[4].height = middleHeight;
    AddQuadIntoMirrorBuffer(mpMirrorBufferDisabled, 5*6, (float)mControlDimensions.x+mSmallestLeftSizeIdle, (float)mControlDimensions.y+mSmallestTopSizeIdle+middleHeight, (float) middleWidth, (float) mpDropdownIdleImageSizeList[5].height, mpDropdownUVCoords_disabled[2*5], mpDropdownUVCoords_disabled[2*5+1] );
    mpDropdownIdleSizeList[5].width = middleWidth;

    // right
    AddQuadIntoMirrorBuffer(mpMirrorBufferDisabled, 6*6, (float)mControlDimensions.x+mSmallestLeftSizeIdle+middleWidth, (float)mControlDimensions.y, (float) mpDropdownIdleImageSizeList[6].width, (float)mpDropdownIdleImageSizeList[6].height, mpDropdownUVCoords_disabled[2*6], mpDropdownUVCoords_disabled[2*6+1] );
    AddQuadIntoMirrorBuffer(mpMirrorBufferDisabled, 7*6, (float)mControlDimensions.x+mSmallestLeftSizeIdle+middleWidth, (float)mControlDimensions.y+mSmallestTopSizeIdle, (float) mpDropdownIdleImageSizeList[7].width, (float)middleHeight, mpDropdownUVCoords_disabled[2*7], mpDropdownUVCoords_disabled[2*7+1] );
    mpDropdownIdleSizeList[7].height = middleHeight;
    AddQuadIntoMirrorBuffer(mpMirrorBufferDisabled, 8*6, (float)mControlDimensions.x+mSmallestLeftSizeIdle+middleWidth, (float)mControlDimensions.y+mSmallestTopSizeIdle+middleHeight, (float) mpDropdownIdleImageSizeList[8].width, (float)mpDropdownIdleImageSizeList[8].height, mpDropdownUVCoords_disabled[2*8], mpDropdownUVCoords_disabled[2*8+1] );

    // 2 buttons (up and pressed) at same location
    CalculateButtonRect(rect);
    AddQuadIntoMirrorBuffer(mpMirrorBufferDisabled,  9*6, (float)rect.x, (float)rect.y, (float) mpDropdownIdleImageSizeList[ 9].width, (float)mpDropdownIdleImageSizeList[ 9].height, mpDropdownUVCoords_disabled[2*9], mpDropdownUVCoords_disabled[2*9+1] );
    AddQuadIntoMirrorBuffer(mpMirrorBufferDisabled, 10*6, (float)rect.x, (float)rect.y, (float) mpDropdownIdleImageSizeList[10].width, (float)mpDropdownIdleImageSizeList[10].height, mpDropdownUVCoords_disabled[2*10], mpDropdownUVCoords_disabled[2*10+1] );   


    // -- now handle the text --

    // draw the selected item in the readout display
    if(-1!=mSelectedItemIndex)
    {
        int x,y;
        CalculateReadoutTextPosition(x,y);
        cString string;
        mpListOfSelectableItems[mSelectedItemIndex]->GetString(string);
        mpSelectedItemCopy->SetText(string);
        mpSelectedItemCopy->SetPosition(x,y);

        mpListOfSelectableItems[mSelectedItemIndex]->SetPosition(x,y);
    }

    // calculate each of the individual dropdown text string locations
    CalculateTrayRect(inner, outer);

    float tx, ty;
    tx = (float)outer.x;
    ty = (float)outer.y;

    int sx = (inner.x + mpDropdownIdleImageSizeList[CLeftMid].width);
    int sy = (inner.y + CPUT_DROPDOWN_TEXT_PADDING);

    for(UINT i=0;i<mpListOfSelectableItems.size();i++)
    {
        mpListOfSelectableItems[i]->SetPosition(sx, sy);

        CPUT_RECT rect;
        mpListOfSelectableItems[i]->GetDimensions(rect.width, rect.height);
        sy+=rect.height + 2*CPUT_DROPDOWN_TEXT_PADDING;
    }

    // mark this as dirty
    mControlGraphicsDirty = true;
}



//--------------------------------------------------------------------------------
void CPUTDropdown::DrawIntoBuffer(CPUTGUIVertex *pVertexBufferMirror, UINT *pInsertIndex, UINT pMaxBufferSize, CPUTGUIVertex *pTextVertexBufferMirror, UINT *pTextInsertIndex, UINT MaxTextVertexBufferSize)
{
    if(!mControlVisible)
    {
        return;
    }

    if((NULL==pVertexBufferMirror) || (NULL==pInsertIndex))
    {
        return;
    }

    if(!mpMirrorBufferActive )
    {
        return;    
    }

    // Do we have enough room to put this control into the output buffer?
    int VertexCopyCount = GetOutputVertexCount();
    ASSERT( (pMaxBufferSize >= *pInsertIndex + VertexCopyCount), _L("Too many CPUT GUI controls for allocated GUI buffer. Allocated GUI vertex buffer is too small.\n\nIncrease CPUT_GUI_BUFFER_SIZE size.") );
    
    switch(mControlState)
    {
    case CPUT_CONTROL_ACTIVE:          
        // copy the standard part of the control (selected box) - first 9 quads
        memcpy(&pVertexBufferMirror[*pInsertIndex], mpMirrorBufferActive, sizeof(CPUTGUIVertex)*6*9);
        *pInsertIndex+= 6*9;

        if((CPUT_DROPDOWN_GUI_MOUSE_PRESSED == mControlGuiState) )//&& (0 != mpListOfSelectableItems.size()))
        {
            // copy the 'down' button
            memcpy(&pVertexBufferMirror[*pInsertIndex], &mpMirrorBufferActive[10*6], sizeof(CPUTGUIVertex)*6*1);
        }
        else
        {
            // copy the 'up' button 
            memcpy(&pVertexBufferMirror[*pInsertIndex], &mpMirrorBufferActive[9*6], sizeof(CPUTGUIVertex)*6*1);
        }
        *pInsertIndex+= 6*1;

        // tray is down, draw it, +1 for the highlit item
        if((CPUT_DROPDOWN_GUI_MOUSE_PRESSED == mControlGuiState) && (0 != mpListOfSelectableItems.size()))
        {
            int QuadsInTray = 6;
            memcpy(&pVertexBufferMirror[*pInsertIndex], &mpMirrorBufferActive[11*6], sizeof(CPUTGUIVertex)*6*(QuadsInTray+1));
            *pInsertIndex+= 6*(QuadsInTray+1);
        }

        break;

    case CPUT_CONTROL_INACTIVE:
        // copy the inactive images into the stream
        memcpy(&pVertexBufferMirror[*pInsertIndex], mpMirrorBufferDisabled, sizeof(CPUTGUIVertex)*6*10);
        *pInsertIndex+= 6*10;
        break;

    default:
        // error! unknown state
        ASSERT(0,_L("CPUTButton: Control is in unknown state"));
        return;
    }


    // -- draw the text --
    // draw selected item in the selection list
    if(NULL!=mpSelectedItemCopy)
    {
        mpSelectedItemCopy->DrawIntoBuffer((CPUTGUIVertex*)pTextVertexBufferMirror, pTextInsertIndex, MaxTextVertexBufferSize);        
    }

    // draw the tray items
    if((CPUT_DROPDOWN_GUI_MOUSE_PRESSED == mControlGuiState) && (0 != mpListOfSelectableItems.size()))
    {
        for(UINT i=0;i<mpListOfSelectableItems.size();i++)
        {
            mpListOfSelectableItems[i]->DrawIntoBuffer((CPUTGUIVertex*)pTextVertexBufferMirror, pTextInsertIndex, MaxTextVertexBufferSize);
        }
    }

    // we'll mark the control as no longer being 'dirty'
    mControlGraphicsDirty = false;
}




// This generates a quad with the supplied coordinates/uv's/etc.
//--------------------------------------------------------------------------------
void CPUTDropdown::AddQuadIntoMirrorBuffer(CPUTGUIVertex *pMirrorBuffer, int index, float x, float y, float w, float h, float3 uv1, float3 uv2 )
{
    CPUTColor4 color;
    color.r = 1.0f;color.g = 1.0f;color.b = 1.0f;color.a = 1.0f;

    pMirrorBuffer[index+0].Pos = float3( x + 0.0f, y + 0.0f, 1.0f);
    pMirrorBuffer[index+0].UV = float2(uv1.x, uv1.y);
    pMirrorBuffer[index+0].Color = color;

    pMirrorBuffer[index+1].Pos = float3( x + w, y + 0.0f, 1.0f);
    pMirrorBuffer[index+1].UV = float2(uv2.x, uv1.y);
    pMirrorBuffer[index+1].Color = color;

    pMirrorBuffer[index+2].Pos = float3( x + 0.0f, y + h, 1.0f);
    pMirrorBuffer[index+2].UV = float2(uv1.x, uv2.y);
    pMirrorBuffer[index+2].Color = color;

    pMirrorBuffer[index+3].Pos = float3( x + w, y + 0.0f, 1.0f);
    pMirrorBuffer[index+3].UV = float2(uv2.x, uv1.y);
    pMirrorBuffer[index+3].Color = color;

    pMirrorBuffer[index+4].Pos = float3( x + w, y + h, 1.0f);
    pMirrorBuffer[index+4].UV = float2(uv2.x, uv2.y);
    pMirrorBuffer[index+4].Color = color;

    pMirrorBuffer[index+5].Pos = float3( x + 0.0f, y +h, 1.0f);
    pMirrorBuffer[index+5].UV = float2(uv1.x, uv2.y);
    pMirrorBuffer[index+5].Color = color;
}

