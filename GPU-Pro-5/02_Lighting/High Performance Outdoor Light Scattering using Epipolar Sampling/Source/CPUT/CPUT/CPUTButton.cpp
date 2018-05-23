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
#include "CPUTButton.h"
#include "CPUTText.h"

// static initializers
bool CPUTButton::mStaticRegistered = false;

// list of resources sizes
CPUT_SIZE CPUTButton::mpButtonIdleImageSizeList[CPUT_NUM_IMAGES_IN_BUTTON] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
CPUT_SIZE CPUTButton::mpButtonPressedImageSizeList[CPUT_NUM_IMAGES_IN_BUTTON] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
CPUT_SIZE CPUTButton::mpButtonDisabledImageSizeList[CPUT_NUM_IMAGES_IN_BUTTON] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

int CPUTButton::mSmallestLeftSizeIdle=0;
int CPUTButton::mSmallestRightSizeIdle=0;
int CPUTButton::mSmallestTopSizeIdle=0;
int CPUTButton::mSmallestBottomSizeIdle=0;

int CPUTButton::mSmallestLeftSizePressed=0;
int CPUTButton::mSmallestRightSizePressed=0;
int CPUTButton::mSmallestTopSizePressed=0;
int CPUTButton::mSmallestBottomSizePressed=0;

int CPUTButton::mSmallestLeftSizeDisabled=0;
int CPUTButton::mSmallestRightSizeDisabled=0;
int CPUTButton::mSmallestTopSizeDisabled=0;
int CPUTButton::mSmallestBottomSizeDisabled=0;

// texture atlas information
float gAtlasWidth = 256.0f;
float gAtlasHeight = 64.0f;

// Texture atlas coordinates of the active-idle button image
int gUVLocations_active[] = { 
    8,8, 15,31,  // lt
    8,31, 15,32, // lm
    8,32, 15,40, // lb

    15,8,  16,31, // mt
    15,31, 16,32, // mm
    15,32, 16,40, // mb

    16,8, 25,31,  // rt
    16,31, 25,32, // rm
    16,32, 25,40  // rb
};

// Texture atlas coordinates of the pressed button image
int gUVLocations_pressed[] = { 
    63,8, 70,31,  // lt
    63,31, 70,32, // lm
    63,32, 70,40, // lb

    70,8,  71,31, // mt
    70,31, 71,32, // mm
    70,32, 71,40, // mb

    71,8, 81,31,  // rt
    71,31, 81,32, // rm
    71,32, 81,40  // rb
};

// Texture atlas coordinates of the disabled button image
int gUVLocations_disabled[] = { 
    36,8, 43,31,  // lt
    36,31, 43,32, // lm
    36,32, 43,40, // lb

    43,8,  45,31, // mt
    43,31, 45,32, // mm
    43,32, 45,40, // mb

    45,8, 55,31,  // rt
    45,31, 55,32, // rm
    45,32, 55,40  // rb
};

float3 mpUVCoords_active[9*2];
float3 mpUVCoords_pressed[9*2];
float3 mpUVCoords_disabled[9*2];


// Constructor
//-----------------------------------------------------------------------------
CPUTButton::CPUTButton(const cString ControlText, CPUTControlID id, CPUTFont *pFont):
    mbMouseInside(false),
    mbStartedClickInside(false),
    mpButtonText(NULL),
    mpMirrorBufferActive(NULL),
    mpMirrorBufferPressed(NULL),
    mpMirrorBufferDisabled(NULL),
    mpFont(pFont)
{
    // initialize the state variables
    InitializeState();

    // save the control ID for callbacks
    mcontrolID = id;

    // save the font to use for text on this button
    mpFont = pFont;

    // set as enabled
    CPUTControl::SetEnable(true);

    // initialize the size lists
    memset(&mpButtonIdleSizeList, 0, sizeof(CPUT_SIZE) * CPUT_NUM_IMAGES_IN_BUTTON);
    memset(&mpButtonPressedSizeList, 0, sizeof(CPUT_SIZE) * CPUT_NUM_IMAGES_IN_BUTTON);
    memset(&mpButtonDisabledSizeList, 0, sizeof(CPUT_SIZE) * CPUT_NUM_IMAGES_IN_BUTTON);

    // set up the per-instance data
    RegisterInstanceResources();

    // set the text on the button and resize it accordingly
    SetText(ControlText);

    // set the default control position
    SetPosition( 0, 0 );

}

// Initial state of the control's member variables
//-----------------------------------------------------------------------------
void CPUTButton::InitializeState()
{
    mcontrolType = CPUT_BUTTON;

    mButtonState = CPUT_BUTTON_NEUTRAL;

    // dimensions
    mButtonDimensions.x=0;
    mButtonDimensions.y=0;
    mButtonDimensions.width=0;
    mButtonDimensions.height=0;
}

// Destructor
//------------------------------------------------------------------------------
CPUTButton::~CPUTButton()
{
    UnRegisterInstanceResources();
}


// Return the upper-left screen coordinate location of this control
//--------------------------------------------------------------------------------
void CPUTButton::GetPosition(int &x, int &y)
{
    x = mButtonDimensions.x;
    y = mButtonDimensions.y;
}

// Return the width/height of the control
//--------------------------------------------------------------------------------
void CPUTButton::GetDimensions(int &width, int &height)
{
    width = mButtonDimensions.width;
    height = mButtonDimensions.height;
}

// Returns the number of quads needed to draw this control
//--------------------------------------------------------------------------------
unsigned int CPUTButton::GetOutputVertexCount()
{
    // A button is always made of 9 quads.
    // Quads 2,4,5,6, and 8 'stretch' in height and/or width to fit the 
    // static text/content inside the button
    //
    //   ---+-----------+---
    //  | 1 |     4     | 7 |              
    //  |---+-----------+---|
    //  |   |           |   | 
    //  | 2 |     5     | 8 |              
    //  |   |           |   | 
    //  |---+-----------+---|
    //  | 3 |     6     | 9 |
    //   ---+-----------+---
    //
    // calculation: 3 verts/triangle * 2 triangle/quad * 9 quads
    return (2*3)*9;
}






//CPUTEventHandler

// Handle keyboard events
//--------------------------------------------------------------------------------
CPUTEventHandledCode CPUTButton::HandleKeyboardEvent(CPUTKey key)
{
    UNREFERENCED_PARAMETER(key);
    return CPUT_EVENT_UNHANDLED;
}

// Handle mouse events
//--------------------------------------------------------------------------------
CPUTEventHandledCode CPUTButton::HandleMouseEvent(int x, int y, int wheel, CPUTMouseState state)
{
    UNREFERENCED_PARAMETER(wheel);
    CPUTEventHandledCode handledCode = CPUT_EVENT_UNHANDLED;

    if((CPUT_CONTROL_INACTIVE == mControlState) || (false == mControlVisible) )
    {
        mbMouseInside = false;
        return handledCode;
    }

    // if we're continuing to be pressed, move around with the mouse movement
    if( (CPUT_BUTTON_PRESSED == mButtonState ) && (CPUT_MOUSE_LEFT_DOWN == state))
    {
        return CPUT_EVENT_HANDLED;
    }

    if(ContainsPoint(x,y))
    {
        // did we start our click inside the button?
        if((state & CPUT_MOUSE_LEFT_DOWN) && (true == mbMouseInside))
        {
            mbStartedClickInside = true;
            mButtonState = CPUT_BUTTON_PRESSED;
            handledCode = CPUT_EVENT_HANDLED;

            // tell gui system this control image is now dirty
            // and needs to rebuild it's draw list
            mControlGraphicsDirty = true;
        }

        // did they click inside the button?
        if(!(state & CPUT_MOUSE_LEFT_DOWN) && (true == mbStartedClickInside) && (CPUT_BUTTON_PRESSED == mButtonState))
        {
            // they let up the click - trigger the user's callback
            mpCallbackHandler->HandleCallbackEvent(1, mcontrolID, (CPUTControl*) this);
            handledCode = CPUT_EVENT_HANDLED;
            mButtonState = CPUT_BUTTON_NEUTRAL;

            // tell gui system this control image is now dirty
            // and needs to rebuild it's draw list
            mControlGraphicsDirty = true;               
        }

        if(!(state & CPUT_MOUSE_LEFT_DOWN))
        {
            mbMouseInside = true;
        }
    }
    else
    {
        // we left the button
        // if we weren't already in neutral state, return to neutral sate
        // this handles case of clicking button, exiting button, and releasing button outside control
        if(CPUT_BUTTON_NEUTRAL != mButtonState)
        {
            mButtonState = CPUT_BUTTON_NEUTRAL;
            mControlGraphicsDirty = true;
        }
        mbMouseInside = false;
        mButtonState = CPUT_BUTTON_NEUTRAL;
        mbStartedClickInside = false;  
    }

    return handledCode;
}

// Returns true if the x,y coordinate is inside the button's control region
//--------------------------------------------------------------------------------
bool CPUTButton::ContainsPoint(int x, int y)
{
    if( (x>=mButtonDimensions.x) && (x<=mButtonDimensions.x+mButtonDimensions.width))
    {
        if( (y>=mButtonDimensions.y) && (y<=mButtonDimensions.y+mButtonDimensions.height))
        {
            return true;
        }
    }
    return false;
}

// Returns the x,y coordinate inside the button area that should be 'safe' to draw on
//--------------------------------------------------------------------------------
void CPUTButton::GetInsetTextCoordinate(int &x, int &y)
{
    // get text size
    CPUT_RECT ButtonTextDimensions;
    if(mpButtonText)
    {
        mpButtonText->GetDimensions(ButtonTextDimensions.width, ButtonTextDimensions.height);
    }
    else
    {
        ButtonTextDimensions.width=0;
        ButtonTextDimensions.height=0;
    }

    // calculate a good 'center' point
    x =(int) ( mButtonDimensions.x + mButtonDimensions.width/2.0f - ButtonTextDimensions.width/2.0f);
    y =(int) ( mButtonDimensions.y + mButtonDimensions.height/2.0f - ButtonTextDimensions.height/2.0f);
}

// Sets the text on the control
//--------------------------------------------------------------------------------
void CPUTButton::SetText(const cString String)
{
    // Zero out the size and location
    InitializeState();

    // create the static text object if it doesn't exist
    if(NULL == mpButtonText)
    {
        mpButtonText = new CPUTText(mpFont);
    }

    // set the Static control's text
    mpButtonText->SetText(String);

    // get the dimensions of the string in pixels
    CPUT_RECT rect;
    mpButtonText->GetDimensions(rect.width, rect.height);

    // resize this control to fix that string with padding
    Resize(rect.width, rect.height);

    // move the text to a nice inset location inside the 'safe' area
    // of the button image
    int x,y;
    GetInsetTextCoordinate(x, y);
    mpButtonText->SetPosition(x, y);

    // position or size may move - force a recalculation of this control's location
    // if it is managed by the auto-arrange function
    if(this->IsAutoArranged())
    {
        mControlNeedsArrangmentResizing = true;
    }
    else
    {
        // otherwise, we mark this as dirty
        mControlGraphicsDirty = true;
    }
}

// sets the dimensions of the button
//--------------------------------------------------------------------------------
void CPUTButton::SetDimensions(int width, int height)
{
    // Zero out the size and location
    InitializeState();

    // get the dimensions of the string in pixels
    CPUT_RECT rect;
    mpButtonText->GetDimensions(rect.width, rect.height);

    width = max(rect.width, width);
    height = max(rect.height, height);

    // resize this control to fix that string with padding
    Resize(width, height);

    // move the text to a nice inset location inside the 'safe' area
    // of the button image
    int x,y;
    GetInsetTextCoordinate(x, y);
    mpButtonText->SetPosition(x, y);
}

// Fills the users buffer with the button text
//--------------------------------------------------------------------------------
void CPUTButton::GetText(cString &String)
{
    if(mpButtonText)
    {
        mpButtonText->GetString(String);
    }
}

// Enable/disable the control
//--------------------------------------------------------------------------------
void CPUTButton::SetEnable(bool in_bEnabled)
{
    // chain to CPUTControl set enabled
    CPUTControl::SetEnable(in_bEnabled);

    // set the control's text to match
    mpButtonText->SetEnable(in_bEnabled);

    // otherwise, we mark this as dirty
    mControlGraphicsDirty = true;
}

// Set the upper-left screen coordinate location of this control
//--------------------------------------------------------------------------------
void CPUTButton::SetPosition(int x, int y)
{
    // move the button graphics
    mButtonDimensions.x = x;
    mButtonDimensions.y = y;

    // move the static text (if any)
    if(mpButtonText)
    {
		// resize things in the buffers
		CPUT_RECT rect;
		mpButtonText->GetDimensions(rect.width, rect.height);
		Resize(rect.width, rect.height);

		int insetX, insetY;
		GetInsetTextCoordinate(insetX, insetY);
		mpButtonText->SetPosition(insetX, insetY);
    }
}

// 'Draw' this control into the supplied vertex buffer object
//--------------------------------------------------------------------------------
void CPUTButton::DrawIntoBuffer(CPUTGUIVertex *pVertexBufferMirror, UINT *pInsertIndex, UINT pMaxBufferSize, CPUTGUIVertex *pTextVertexBufferMirror, UINT *pTextInsertIndex, UINT MaxTextVertexBufferSize)
{
    if(!mControlVisible)
    {
        return;
    }

    // invalid output buffer pointers?
    if((NULL==pVertexBufferMirror) || (NULL==pInsertIndex))
    {
        return;
    }

    // invalid buffer pointers?
    if(!mpMirrorBufferActive || !mpMirrorBufferPressed || !mpMirrorBufferDisabled)
    {
        return;
    }

    // Do we have enough room to put this control into the output buffer?
    int VertexCopyCount = GetOutputVertexCount();
    ASSERT( (pMaxBufferSize >= *pInsertIndex + VertexCopyCount), _L("Too many CPUT GUI controls for allocated GUI buffer. Allocated GUI vertex buffer is too small.\n\nIncrease CPUT_GUI_BUFFER_SIZE size.") );
    
    switch(mControlState)
    {
    case CPUT_CONTROL_ACTIVE:
        // copy the active+idle button into the stream
        if(CPUT_BUTTON_NEUTRAL == mButtonState)
        {
            memcpy(&pVertexBufferMirror[*pInsertIndex], mpMirrorBufferActive, sizeof(CPUTGUIVertex)*VertexCopyCount);
        }        

        // copy the pressed button into the stream
        if(CPUT_BUTTON_PRESSED == mButtonState)
        {
            memcpy(&pVertexBufferMirror[*pInsertIndex], mpMirrorBufferPressed, sizeof(CPUTGUIVertex)*VertexCopyCount);
        }
        
        break;
    case CPUT_CONTROL_INACTIVE:
        // copy the inactive button into the stream
        memcpy(&pVertexBufferMirror[*pInsertIndex], mpMirrorBufferDisabled, sizeof(CPUTGUIVertex)*VertexCopyCount);
        break;

    default:
        // error! unknown state
        ASSERT(0,_L("CPUTButton: Control is in unknown state"));
        return;
    }

    // move the index the correct number of floats to account
    // for 9 new quads, each quad with 6 verts in it (and each vert with 3+2 floats in it).
    *pInsertIndex+= VertexCopyCount;

    // now do the text
    // draw the text
    if(mpButtonText)
    {
        mpButtonText->DrawIntoBuffer(pTextVertexBufferMirror, pTextInsertIndex, MaxTextVertexBufferSize);
    }

    // we'll mark the control as no longer being 'dirty'
    mControlGraphicsDirty = false;
}


// Allocates/registers resources used by all buttons
//--------------------------------------------------------------------------------
CPUTResult CPUTButton::RegisterStaticResources()
{
    // calculate the UV coordinates of each of the 9 images that
    // make up a button.  Do this for the active, pressed, and disabled states.
    for(int ii=0; ii<18; ii++)
    {
        mpUVCoords_active[ii].x = gUVLocations_active[2*ii]/gAtlasWidth;
        mpUVCoords_active[ii].y = gUVLocations_active[2*ii+1]/gAtlasHeight;

        mpUVCoords_pressed[ii].x = gUVLocations_pressed[2*ii]/gAtlasWidth;
        mpUVCoords_pressed[ii].y = gUVLocations_pressed[2*ii+1]/gAtlasHeight;

        mpUVCoords_disabled[ii].x = gUVLocations_disabled[2*ii]/gAtlasWidth;
        mpUVCoords_disabled[ii].y = gUVLocations_disabled[2*ii+1]/gAtlasHeight;
    }

    // calculate the width/height in pixels of each of the 9 image slices
    // that makes up the button images
    int QuadIndex=0;
    for(int ii=0; ii<9*4; ii+=4)
    {
        mpButtonIdleImageSizeList[QuadIndex].width = gUVLocations_active[ii+2] - gUVLocations_active[ii+0];
        mpButtonIdleImageSizeList[QuadIndex].height = gUVLocations_active[ii+3] - gUVLocations_active[ii+1];
 
        mpButtonPressedImageSizeList[QuadIndex].width = gUVLocations_pressed[ii+2] - gUVLocations_pressed[ii+0];
        mpButtonPressedImageSizeList[QuadIndex].height = gUVLocations_pressed[ii+3] - gUVLocations_pressed[ii+1];

        mpButtonDisabledImageSizeList[QuadIndex].width = gUVLocations_disabled[ii+2] - gUVLocations_disabled[ii+0];
        mpButtonDisabledImageSizeList[QuadIndex].height = gUVLocations_disabled[ii+3] - gUVLocations_disabled[ii+1];
        QuadIndex++;
    }
    
    // find the narrowest (width) left side images
    mSmallestLeftSizeIdle = min(min(mpButtonIdleImageSizeList[0].width, mpButtonIdleImageSizeList[1].width), mpButtonIdleImageSizeList[2].width);
    mSmallestLeftSizePressed = min(min(mpButtonPressedImageSizeList[0].width, mpButtonPressedImageSizeList[1].width), mpButtonPressedImageSizeList[2].width);
    mSmallestLeftSizeDisabled = min(min(mpButtonDisabledImageSizeList[0].width, mpButtonDisabledImageSizeList[1].width), mpButtonDisabledImageSizeList[2].width);  
    
    // find the narrowest (width) right side images
    mSmallestRightSizeIdle = min(min(mpButtonIdleImageSizeList[6].width, mpButtonIdleImageSizeList[7].width), mpButtonIdleImageSizeList[8].width);
    mSmallestRightSizePressed = min(min(mpButtonPressedImageSizeList[6].width, mpButtonPressedImageSizeList[7].width), mpButtonPressedImageSizeList[8].width);
    mSmallestRightSizeDisabled = min(min(mpButtonDisabledImageSizeList[6].width, mpButtonDisabledImageSizeList[7].width), mpButtonDisabledImageSizeList[8].width);  
    

    // find the shortest (height) of the top row of images
    mSmallestTopSizeIdle = min(min( mpButtonIdleImageSizeList[0].height,mpButtonIdleImageSizeList[3].height), mpButtonIdleImageSizeList[6].height);
    mSmallestTopSizePressed = min(min( mpButtonPressedImageSizeList[0].height,mpButtonPressedImageSizeList[3].height), mpButtonPressedImageSizeList[6].height);
    mSmallestTopSizeDisabled = min( min( mpButtonDisabledImageSizeList[0].height,mpButtonDisabledImageSizeList[3].height), mpButtonDisabledImageSizeList[6].height);

    // find the shortest (height) of the bottom row of images
    mSmallestBottomSizeIdle = min(min( mpButtonIdleImageSizeList[2].height,mpButtonIdleImageSizeList[5].height), mpButtonIdleImageSizeList[8].height);
    mSmallestBottomSizePressed = min(min( mpButtonPressedImageSizeList[2].height,mpButtonPressedImageSizeList[5].height), mpButtonPressedImageSizeList[8].height);
    mSmallestBottomSizeDisabled = min(min( mpButtonDisabledImageSizeList[2].height,mpButtonDisabledImageSizeList[5].height), mpButtonDisabledImageSizeList[8].height);
        
    mStaticRegistered = true;

    return CPUT_SUCCESS;
}


// Deletes any statically allocated resources used for all buttons
//--------------------------------------------------------------------------------
CPUTResult CPUTButton::UnRegisterStaticResources()
{
    return CPUT_SUCCESS;
}

// Allocates an initialize all per-instance resources for this button
//--------------------------------------------------------------------------------
CPUTResult CPUTButton::RegisterInstanceResources()
{
    // clear any previously allocated buffers
    SAFE_DELETE_ARRAY(mpMirrorBufferActive);
    SAFE_DELETE_ARRAY(mpMirrorBufferPressed);
    SAFE_DELETE_ARRAY(mpMirrorBufferDisabled);

    // allocate the per-instance sizes (each button will have different dimensions)
    mpMirrorBufferActive = new CPUTGUIVertex[6 * 9];
    mpMirrorBufferPressed = new CPUTGUIVertex[6 * 9];
    mpMirrorBufferDisabled = new CPUTGUIVertex[6 * 9];


    // store all the default button component quad sizes in instance variables
    // Re-sizable parts will get re-calculated during setText and other operations
    for(int i=0; i<CPUT_NUM_IMAGES_IN_BUTTON; i++)
    {
        mpButtonIdleSizeList[i].height = mpButtonIdleImageSizeList[i].height;
        mpButtonIdleSizeList[i].width = mpButtonIdleImageSizeList[i].width;

        mpButtonPressedSizeList[i].height = mpButtonPressedImageSizeList[i].height;
        mpButtonPressedSizeList[i].width = mpButtonPressedImageSizeList[i].width;

        mpButtonDisabledSizeList[i].height = mpButtonDisabledImageSizeList[i].height;
        mpButtonDisabledSizeList[i].width = mpButtonDisabledImageSizeList[i].width;
    }

    return CPUT_SUCCESS;
}

// Delete all instance resources alloated for this button
//--------------------------------------------------------------------------------
CPUTResult CPUTButton::UnRegisterInstanceResources()
{
    CPUTResult result = CPUT_SUCCESS;

    // delete the static text object
    SAFE_DELETE(mpButtonText);

    // Release the mirrored vertex lists
    SAFE_DELETE_ARRAY(mpMirrorBufferActive);
    SAFE_DELETE_ARRAY(mpMirrorBufferPressed);
    SAFE_DELETE_ARRAY(mpMirrorBufferDisabled);

    return result;
}



// Resize the button 
// Recalculates the size of the button based on the supplied dimensions and 
// generates new vertex buffer lists for each of the 3 states of the control.
//
// It does NOT move the inside text, nor does it base itself on the size of the 
// text. You need to do that yourself outside this function and pass it in
//--------------------------------------------------------------------------------
CPUTResult CPUTButton::Resize(int width, int height)
{
    // verify that the new dimensions fit the minimal 'safe' dimensions needed to draw the button
    // or ugly clipping will occur
    int safeWidth=0;
    int safeHeight=0;

    switch(mControlState)
    {
    case CPUT_CONTROL_ACTIVE:
        if(CPUT_BUTTON_NEUTRAL == mButtonState)
        {
            safeWidth =  mSmallestLeftSizeIdle + mSmallestRightSizeIdle + 1;
            safeHeight =  mSmallestTopSizeIdle + mSmallestBottomSizeIdle + 1;
        }
        if(CPUT_BUTTON_PRESSED == mButtonState)
        {
            safeWidth =  mSmallestLeftSizePressed + mSmallestRightSizePressed + 1;
            safeHeight =  mSmallestTopSizePressed + mSmallestBottomSizePressed + 1;
        }
        break;

    case CPUT_CONTROL_INACTIVE:        
        safeWidth =  mSmallestLeftSizeDisabled + mSmallestRightSizeDisabled + 1;
        safeHeight =  mSmallestTopSizeDisabled + mSmallestBottomSizeDisabled + 1;
        break;

    default:
        ASSERT(0,_L("")); // todo: error! unknown state - using idle dimensions as a default
        safeWidth =  mSmallestLeftSizeIdle + mSmallestRightSizeIdle + 1;
        safeHeight =  mSmallestTopSizeIdle + mSmallestBottomSizeIdle + 1;
    }

    // if the user's dimensions are smaller than the smallest 'safe' dimensions of the button,
    // use the safe ones instead.
    if(safeWidth > width)
    {
        width = safeWidth;
    }
    if(safeHeight > height)
    {
        height = safeHeight;
    }

    // add some padding for nicety
    width += CPUT_BUTTON_TEXT_BORDER_PADDING_X;
    height += CPUT_BUTTON_TEXT_BORDER_PADDING_Y;
  
    {
        // store the new dimensions
        mButtonDimensions.width = width;
        mButtonDimensions.height = height;

        // calculate the pieces we'll need to rebuild
        int middleWidth = width - mSmallestLeftSizeIdle - mSmallestRightSizeIdle;
        int middleHeight = height - mSmallestTopSizeIdle - mSmallestBottomSizeIdle;

        // delete the old button quads for the middle sections
        //result = UnRegisterResizableInstanceQuads();

        // create a new quads with the correct size
        // Idle button quads        
        
        // left
        AddQuadIntoMirrorBuffer(mpMirrorBufferActive, 0*6, (float)mButtonDimensions.x, (float)mButtonDimensions.y, (float) mpButtonIdleSizeList[0].width, (float) mpButtonIdleSizeList[0].height, mpUVCoords_active[2*0], mpUVCoords_active[2*0+1] );
        AddQuadIntoMirrorBuffer(mpMirrorBufferActive, 1*6, (float)mButtonDimensions.x, (float)mButtonDimensions.y+mSmallestTopSizeIdle, (float) mpButtonIdleSizeList[1].width, (float) middleHeight, mpUVCoords_active[2*1], mpUVCoords_active[2*1+1] );
        mpButtonIdleSizeList[1].height = middleHeight;
        AddQuadIntoMirrorBuffer(mpMirrorBufferActive, 2*6, (float)mButtonDimensions.x, (float)mButtonDimensions.y+mSmallestTopSizeIdle+middleHeight, (float) mpButtonIdleSizeList[2].width, (float) mpButtonIdleSizeList[2].height, mpUVCoords_active[2*2], mpUVCoords_active[2*2+1] );

        // middle
        AddQuadIntoMirrorBuffer(mpMirrorBufferActive, 3*6, (float)mButtonDimensions.x+mSmallestLeftSizeIdle, (float)mButtonDimensions.y, (float) middleWidth, (float) mpButtonIdleSizeList[3].height, mpUVCoords_active[2*3], mpUVCoords_active[2*3+1] );
        mpButtonIdleSizeList[3].width = middleWidth;
        AddQuadIntoMirrorBuffer(mpMirrorBufferActive, 4*6, (float)mButtonDimensions.x+mSmallestLeftSizeIdle, (float)mButtonDimensions.y+mSmallestTopSizeIdle, (float) middleWidth, (float) middleHeight, mpUVCoords_active[2*4], mpUVCoords_active[2*4+1] );
        mpButtonIdleSizeList[4].width = middleWidth;
        mpButtonIdleSizeList[4].height = middleHeight;
        AddQuadIntoMirrorBuffer(mpMirrorBufferActive, 5*6, (float)mButtonDimensions.x+mSmallestLeftSizeIdle, (float)mButtonDimensions.y+mSmallestTopSizeIdle+middleHeight, (float) middleWidth, (float) mpButtonIdleSizeList[5].height, mpUVCoords_active[2*5], mpUVCoords_active[2*5+1] );
        mpButtonIdleSizeList[5].width = middleWidth;

        // right
        AddQuadIntoMirrorBuffer(mpMirrorBufferActive, 6*6, (float)mButtonDimensions.x+mSmallestLeftSizeIdle+middleWidth, (float)mButtonDimensions.y, (float) mpButtonIdleSizeList[6].width, (float)mpButtonIdleSizeList[6].height, mpUVCoords_active[2*6], mpUVCoords_active[2*6+1] );
        AddQuadIntoMirrorBuffer(mpMirrorBufferActive, 7*6, (float)mButtonDimensions.x+mSmallestLeftSizeIdle+middleWidth, (float)mButtonDimensions.y+mSmallestTopSizeIdle, (float) mpButtonIdleSizeList[7].width, (float)middleHeight, mpUVCoords_active[2*7], mpUVCoords_active[2*7+1] );
        mpButtonIdleSizeList[7].height = middleHeight;
        AddQuadIntoMirrorBuffer(mpMirrorBufferActive, 8*6, (float)mButtonDimensions.x+mSmallestLeftSizeIdle+middleWidth, (float)mButtonDimensions.y+mSmallestTopSizeIdle+middleHeight, (float) mpButtonIdleSizeList[8].width, (float)mpButtonIdleSizeList[8].height, mpUVCoords_active[2*8], mpUVCoords_active[2*8+1] );

        // register uberbuffer
        //RegisterUberBuffer(pImmediateContext, &mpUberBufferActive, mpMirrorBufferActive);

        
        // Pressed button quads
        // left
        AddQuadIntoMirrorBuffer(mpMirrorBufferPressed, 0*6, (float)mButtonDimensions.x, (float)mButtonDimensions.y, (float) mpButtonPressedSizeList[0].width, (float) mpButtonPressedSizeList[0].height, mpUVCoords_pressed[2*0], mpUVCoords_pressed[2*0+1] );
        AddQuadIntoMirrorBuffer(mpMirrorBufferPressed, 1*6, (float)mButtonDimensions.x, (float)mButtonDimensions.y+mSmallestTopSizePressed, (float) mpButtonPressedSizeList[1].width, (float) middleHeight, mpUVCoords_pressed[2*1], mpUVCoords_pressed[2*1+1] );
        mpButtonPressedSizeList[1].height = middleHeight;
        AddQuadIntoMirrorBuffer(mpMirrorBufferPressed, 2*6, (float)mButtonDimensions.x, (float)mButtonDimensions.y+mSmallestTopSizePressed+middleHeight, (float) mpButtonPressedSizeList[2].width, (float) mpButtonPressedSizeList[2].height, mpUVCoords_pressed[2*2], mpUVCoords_pressed[2*2+1] );

        // middle
        AddQuadIntoMirrorBuffer(mpMirrorBufferPressed, 3*6, (float)mButtonDimensions.x+mSmallestLeftSizePressed, (float)mButtonDimensions.y, (float) middleWidth, (float) mpButtonPressedSizeList[3].height, mpUVCoords_pressed[2*3], mpUVCoords_pressed[2*3+1] );
        mpButtonPressedSizeList[3].width = middleWidth;
        AddQuadIntoMirrorBuffer(mpMirrorBufferPressed, 4*6, (float)mButtonDimensions.x+mSmallestLeftSizePressed, (float)mButtonDimensions.y+mSmallestTopSizePressed, (float) middleWidth, (float) middleHeight, mpUVCoords_pressed[2*4], mpUVCoords_pressed[2*4+1] );
        mpButtonPressedSizeList[4].width = middleWidth;
        mpButtonPressedSizeList[4].height = middleHeight;        
        AddQuadIntoMirrorBuffer(mpMirrorBufferPressed, 5*6, (float)mButtonDimensions.x+mSmallestLeftSizePressed, (float)mButtonDimensions.y+mSmallestTopSizePressed+middleHeight, (float) middleWidth, (float) mpButtonPressedSizeList[5].height, mpUVCoords_pressed[2*5], mpUVCoords_pressed[2*5+1] );
        mpButtonPressedSizeList[5].width = middleWidth;

        // right
        AddQuadIntoMirrorBuffer(mpMirrorBufferPressed, 6*6, (float)mButtonDimensions.x+mSmallestLeftSizePressed+middleWidth, (float)mButtonDimensions.y, (float) mpButtonPressedSizeList[6].width, (float) mpButtonPressedSizeList[6].height, mpUVCoords_pressed[2*6], mpUVCoords_pressed[2*6+1] );
        AddQuadIntoMirrorBuffer(mpMirrorBufferPressed, 7*6, (float)mButtonDimensions.x+mSmallestLeftSizePressed+middleWidth, (float)mButtonDimensions.y+mSmallestTopSizePressed, (float) mpButtonPressedSizeList[7].width, (float) middleHeight, mpUVCoords_pressed[2*7], mpUVCoords_pressed[2*7+1] );
        mpButtonPressedSizeList[7].height = middleHeight;
        AddQuadIntoMirrorBuffer(mpMirrorBufferPressed, 8*6, (float)mButtonDimensions.x+mSmallestLeftSizePressed+middleWidth, (float)mButtonDimensions.y+mSmallestTopSizePressed+middleHeight, (float) mpButtonPressedSizeList[8].width, (float) mpButtonPressedSizeList[8].height, mpUVCoords_pressed[2*8], mpUVCoords_pressed[2*8+1] );


        // Disabled button quads
        // left
        AddQuadIntoMirrorBuffer(mpMirrorBufferDisabled, 0*6, (float)mButtonDimensions.x, (float)mButtonDimensions.y, (float) mpButtonDisabledSizeList[0].width, (float) mpButtonDisabledSizeList[0].height, mpUVCoords_disabled[2*0], mpUVCoords_disabled[2*0+1] );
        AddQuadIntoMirrorBuffer(mpMirrorBufferDisabled, 1*6, (float)mButtonDimensions.x, (float)mButtonDimensions.y+mSmallestTopSizePressed, (float) mpButtonDisabledSizeList[1].width, (float) middleHeight, mpUVCoords_disabled[2*1], mpUVCoords_disabled[2*1+1] );
        mpButtonDisabledSizeList[1].height = middleHeight;
        AddQuadIntoMirrorBuffer(mpMirrorBufferDisabled, 2*6, (float)mButtonDimensions.x, (float)mButtonDimensions.y+mSmallestTopSizePressed+middleHeight, (float) mpButtonDisabledSizeList[2].width, (float) mpButtonDisabledSizeList[2].height, mpUVCoords_disabled[2*2], mpUVCoords_disabled[2*2+1] );

        // middle
        AddQuadIntoMirrorBuffer(mpMirrorBufferDisabled, 3*6, (float)mButtonDimensions.x+mSmallestLeftSizePressed, (float)mButtonDimensions.y, (float) middleWidth, (float) mpButtonDisabledSizeList[3].height, mpUVCoords_disabled[2*3], mpUVCoords_disabled[2*3+1] );
        mpButtonDisabledSizeList[3].width = middleWidth;
        AddQuadIntoMirrorBuffer(mpMirrorBufferDisabled, 4*6, (float)mButtonDimensions.x+mSmallestLeftSizePressed, (float)mButtonDimensions.y+mSmallestTopSizePressed, (float) middleWidth, (float) middleHeight, mpUVCoords_disabled[2*4], mpUVCoords_disabled[2*4+1] );
        mpButtonDisabledSizeList[4].width = middleWidth;
        mpButtonDisabledSizeList[4].height = middleHeight;        
        AddQuadIntoMirrorBuffer(mpMirrorBufferDisabled, 5*6, (float)mButtonDimensions.x+mSmallestLeftSizePressed, (float)mButtonDimensions.y+mSmallestTopSizePressed+middleHeight, (float) middleWidth, (float) mpButtonDisabledSizeList[5].height, mpUVCoords_disabled[2*5], mpUVCoords_disabled[2*5+1] );
        mpButtonDisabledSizeList[5].width = middleWidth;

        // right
        AddQuadIntoMirrorBuffer(mpMirrorBufferDisabled, 6*6, (float)mButtonDimensions.x+mSmallestLeftSizePressed+middleWidth, (float)mButtonDimensions.y, (float) mpButtonDisabledSizeList[6].width, (float) mpButtonDisabledSizeList[6].height, mpUVCoords_disabled[2*6], mpUVCoords_disabled[2*6+1] );
        AddQuadIntoMirrorBuffer(mpMirrorBufferDisabled, 7*6, (float)mButtonDimensions.x+mSmallestLeftSizePressed+middleWidth, (float)mButtonDimensions.y+mSmallestTopSizePressed, (float) mpButtonDisabledSizeList[7].width, (float) middleHeight, mpUVCoords_disabled[2*7], mpUVCoords_disabled[2*7+1] );
        mpButtonDisabledSizeList[7].height = middleHeight;
        AddQuadIntoMirrorBuffer(mpMirrorBufferDisabled, 8*6, (float)mButtonDimensions.x+mSmallestLeftSizePressed+middleWidth, (float)mButtonDimensions.y+mSmallestTopSizePressed+middleHeight, (float) mpButtonDisabledSizeList[8].width, (float) mpButtonDisabledSizeList[8].height, mpUVCoords_disabled[2*8], mpUVCoords_disabled[2*8+1] );
                        
        // Mark this control as 'dirty' for drawing and inform the gui system that
        // it needs to re-calculate it's drawing buffer
        mControlGraphicsDirty = true;        
    }

    return CPUT_SUCCESS;
}



// This generates a quad with the supplied coordinates/uv's/etc.
//------------------------------------------------------------------------
void CPUTButton::AddQuadIntoMirrorBuffer(CPUTGUIVertex *pMirrorBuffer, 
    int index, 
    float x, 
    float y, 
    float w, 
    float h, 
    float3 uv1, 
    float3 uv2 )
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

