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
#include "CPUTSlider.h"
#include "CPUTText.h"
#include <math.h> // for fmod
#include <string.h>

// static initializers
CPUT_SIZE CPUTSlider::mpActiveImageSizeList[] = { {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, };
CPUT_SIZE CPUTSlider::mpPressedImageSizeList[] = { {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, };
CPUT_SIZE CPUTSlider::mpDisabledImageSizeList[] = { {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, };

// texture atlas information
float gAtlasWidthSlider = 256.0f;
float gAtlasHeightSlider= 64.0f;

// Texture atlas coordinates of the active-idle button image
int gUVLocationsSlider_active[] = { 
    141,0, 152,18,  // neutral nub    
    170,5, 171,9,   // tick 
    175,7, 178,11,  // left tray end
    183,7, 184,11,  // tray 
    188,7, 190,11,  // right tray end
};

// Texture atlas coordinates of the pressed button image
int gUVLocationsSlider_pressed[] = { 
    154,0, 165,18,  // grabbed nub
    170,5, 171,9,   // tick 
    175,7, 178,11,  // left tray end
    183,7, 184,11,  // tray 
    188,7, 190,11,  // right tray end
};

// Texture atlas coordinates of the disabled button image
int gUVLocationsSlider_disabled[] = { 
    193,0, 204,18,  // neutral nub    
    217,7, 218,11,   // tick
    206,7, 209,11,  // left tray end
    211,7, 212,11,  // tray 
    214,7, 216,11,  // right tray end 
};

float3 mpUVCoordsSlider_active[10];
float3 mpUVCoordsSlider_pressed[10];
float3 mpUVCoordsSlider_disabled[10];

const int Grip = 0;
const int Tick = 1;
const int TrayLeftCap = 2;
const int Tray = 3;
const int TrayRightCap = 4;

const int NumQuadsinSlider = 5;


// Constructor
//------------------------------------------------------------------------------
CPUTSlider::CPUTSlider(const cString ControlText, CPUTControlID id, CPUTFont *pFont):    mpControlText(NULL),
    mpMirrorBufferActive(NULL),
    mpMirrorBufferPressed(NULL),
    mpMirrorBufferDisabled(NULL),
    mSliderNubTickLocation(0),
    mSliderNubLocation(0.0f),
    mbMouseInside(false),
    mbStartedClickInside(false),
    mbDrawTicks(true),
    mpFont(pFont)
{
    // initialize the state variables
    InitialStateSet();

    // save the control ID for callbacks
    mcontrolID = id;

    // set as enabled
    CPUTControl::SetEnable(true);

    // default to an active control
    mControlState = CPUT_CONTROL_ACTIVE;

    // save the control ID
    mcontrolID = id;

    // set the string to display with the slider
    mpControlText = new CPUTText(mpFont);
    mpControlText->SetText(ControlText);
   
    // set default scale/start/end values
    SetScale(mSliderStartValue, mSliderEndValue, mSliderNumberOfSteps);

}

// Initial state of the control's member variables
//------------------------------------------------------------------------------
void CPUTSlider::InitialStateSet()
{
    mcontrolType = CPUT_SLIDER;
    mControlState = CPUT_CONTROL_ACTIVE;
    mSliderState = CPUT_SLIDER_NIB_UNPRESSED;

    mControlDimensions.x=0; mControlDimensions.y=0; mControlDimensions.width=0; mControlDimensions.height=0;

    // slider state/range
    mSliderStartValue = 1.0f;
    mSliderEndValue = 10.0f;
    mSliderNumberOfSteps = 10;
    mSliderNumberOfTicks = mSliderNumberOfSteps;
    mCurrentSliderValue = 1;
}

// destructor
//------------------------------------------------------------------------------
CPUTSlider::~CPUTSlider()
{
    UnRegisterInstanceResources();
}

//CPUTSlider

// Returns the number of quads needed to draw this control
//--------------------------------------------------------------------------------
unsigned int CPUTSlider::GetOutputVertexCount()
{
    // The base slider is made of:
    // 3 quads for the sliding 'tray'
    // 1 for the nub
    // 1 quad for each slider tick displayed
    //
    //   ---+------------------+---
    //  | 0 |         1        | 2 |              
    //   ---+------------------+--- 

    // Calculation:
    // 3 verticies/triangle * 2 triangles/quad * (3 + 1 + (1 * number of slider ticks) )
    if(false == mbDrawTicks)
    {
        return (3 * 2) * (3 + 1); 
    }
    return ( 3 * 2 ) * (3 + 1 + (1 * mSliderNumberOfTicks)); 
}




//
//--------------------------------------------------------------------------------
void CPUTSlider::SetPosition(int x, int y)
{
    mControlDimensions.x = x;
    mControlDimensions.y = y;

    Recalculate();

    // text is always in the upper-left corner of the control
    mpControlText->SetPosition(x,y);
}

// Register all resources shared by every slider control
//--------------------------------------------------------------------------------
CPUTResult CPUTSlider::RegisterStaticResources()
{    
    // calculate the UV coordinates of each of the images that
    // make up the slider.  Do this for the active, pressed, and disabled states.
    for(int ii=0; ii<10; ii++)
    {
        mpUVCoordsSlider_active[ii].x = gUVLocationsSlider_active[2*ii]/gAtlasWidthSlider;
        mpUVCoordsSlider_active[ii].y = gUVLocationsSlider_active[2*ii+1]/gAtlasHeightSlider;

        mpUVCoordsSlider_pressed[ii].x = gUVLocationsSlider_pressed[2*ii]/gAtlasWidthSlider;
        mpUVCoordsSlider_pressed[ii].y = gUVLocationsSlider_pressed[2*ii+1]/gAtlasHeightSlider;

        mpUVCoordsSlider_disabled[ii].x = gUVLocationsSlider_disabled[2*ii]/gAtlasWidthSlider;
        mpUVCoordsSlider_disabled[ii].y = gUVLocationsSlider_disabled[2*ii+1]/gAtlasHeightSlider;
    }
    

    // calculate the width/height in pixels of each of the 5 image slices
    // that makes up the slider
    int QuadIndex=0;
    for(int ii=0; ii<NumQuadsinSlider*4; ii+=4)
    {
        mpActiveImageSizeList[QuadIndex].width = gUVLocationsSlider_active[ii+2] - gUVLocationsSlider_active[ii+0];
        mpActiveImageSizeList[QuadIndex].height = gUVLocationsSlider_active[ii+3] - gUVLocationsSlider_active[ii+1];
        
        mpPressedImageSizeList[QuadIndex].width = gUVLocationsSlider_pressed[ii+2] - gUVLocationsSlider_pressed[ii+0];
        mpPressedImageSizeList[QuadIndex].height = gUVLocationsSlider_pressed[ii+3] - gUVLocationsSlider_pressed[ii+1];

        mpDisabledImageSizeList[QuadIndex].width = gUVLocationsSlider_disabled[ii+2] - gUVLocationsSlider_disabled[ii+0];
        mpDisabledImageSizeList[QuadIndex].height = gUVLocationsSlider_disabled[ii+3] - gUVLocationsSlider_disabled[ii+1];
        QuadIndex++;
    }

    return CPUT_SUCCESS;
}

// Unregister all the shared slider controls resources
// only call this when NO more slider controls are left
//--------------------------------------------------------------------------------
CPUTResult CPUTSlider::UnRegisterStaticResources()
{
    return CPUT_SUCCESS;
}



//
//--------------------------------------------------------------------------------
CPUTResult CPUTSlider::RegisterInstanceResources()
{
    // clear any previously allocated buffers
    SAFE_DELETE_ARRAY(mpMirrorBufferActive);
    SAFE_DELETE_ARRAY(mpMirrorBufferPressed);
    SAFE_DELETE_ARRAY(mpMirrorBufferDisabled);

    // allocate the per-instance sizes (each button will have different dimensions)
    mpMirrorBufferActive = new CPUTGUIVertex[6*(NumQuadsinSlider+mSliderNumberOfSteps)]; // 50 tick marks
    mpMirrorBufferPressed = new CPUTGUIVertex[6*(NumQuadsinSlider+mSliderNumberOfSteps)];
    mpMirrorBufferDisabled = new CPUTGUIVertex[6*(NumQuadsinSlider+mSliderNumberOfSteps)]; 

    return CPUT_SUCCESS;
}

//
//--------------------------------------------------------------------------------
CPUTResult CPUTSlider::UnRegisterInstanceResources()
{
    // delete the static text object
    SAFE_DELETE(mpControlText);
    
    // clear vertex mirror buffers
    SAFE_DELETE_ARRAY(mpMirrorBufferActive);
    SAFE_DELETE_ARRAY(mpMirrorBufferPressed);
    SAFE_DELETE_ARRAY(mpMirrorBufferDisabled);

    return CPUT_SUCCESS;
}

// Handle mouse events
//--------------------------------------------------------------------------------
CPUTEventHandledCode CPUTSlider::HandleMouseEvent(int x, int y, int wheel, CPUTMouseState state)
{
    CPUTEventHandledCode handledCode = CPUT_EVENT_UNHANDLED;
    UNREFERENCED_PARAMETER(wheel);

    // just return if the control is disabled or invisible
    if((CPUT_CONTROL_INACTIVE == mControlState) || (false == mControlVisible) )
    {
        mbMouseInside = false;
        mbStartedClickInside = false;
        return handledCode;
    }

    // if we're continuing to be pressed, move around with the mouse movement
    //CPUT_CONTROL_PRESSED == mControlState
    if( (CPUT_SLIDER_NIB_PRESSED == mSliderState) && (CPUT_MOUSE_LEFT_DOWN == state) && (true == mbStartedClickInside ))
    {
        mSliderNubLocation = (float)x-mControlDimensions.x-mpActiveImageSizeList[Grip].width/2;

        // are they dragging off the left side?
        if(mSliderNubLocation < 0.0f)
        {
            mSliderNubLocation = 0.0f;
        }

        // are they dragging off the right side?
        float TrayIndent = mpActiveImageSizeList[Grip].width / 2.0f;
        float ExtentX = (float)(TrayIndent + mpActiveImageSizeList[TrayLeftCap].width + CPUT_DEFAULT_TRAY_WIDTH + mpActiveImageSizeList[TrayRightCap].width - mpActiveImageSizeList[Grip].width+2);

        if(mSliderNubLocation > ExtentX)
        {
            mSliderNubLocation = ExtentX;
        }

        // recalculate the location of the items
        Recalculate();


        handledCode = CPUT_EVENT_HANDLED;
        return handledCode;
    }

    // did the slider just get released?
    if( (CPUT_SLIDER_NIB_PRESSED == mSliderState) && (CPUT_MOUSE_LEFT_DOWN != state) && (true == mbStartedClickInside ))
    {
        //mControlState = CPUT_CONTROL_ACTIVE;
        mSliderState = CPUT_SLIDER_NIB_UNPRESSED;
        SnapToNearestTick();
        handledCode = CPUT_EVENT_HANDLED;

        // trigger the user's callback
        mpCallbackHandler->HandleCallbackEvent(1, mcontrolID, (CPUTControl*) this);

        // recalculate the location of the items
        Recalculate();

        // reset flags
        mbStartedClickInside = false;
        return CPUT_EVENT_HANDLED;
    }

    // handle the initial grabbing of the nub
    if(ContainsPoint(x,y))
    {
        handledCode = CPUT_EVENT_HANDLED;

        if(CPUT_MOUSE_LEFT_DOWN == state)
        {
            if(!PointingAtNub(x,y))
            {
                mbMouseInside = false;
                mbStartedClickInside = false;
                //mControlState = CPUT_CONTROL_ACTIVE;
                mSliderState = CPUT_SLIDER_NIB_UNPRESSED;
                handledCode = CPUT_EVENT_UNHANDLED;
            }
            else
            {
                if(true == mbMouseInside)
                {
                    mbStartedClickInside = true;
                    //mControlState = CPUT_CONTROL_PRESSED;
                    mSliderState =  CPUT_SLIDER_NIB_PRESSED;
                }
                else
                {
                    // ignore
                    return CPUT_EVENT_UNHANDLED;
                }
            }
        }
        else
        {
            if(PointingAtNub(x,y))
            {
                mbMouseInside = true;
            }
            else
            {
                //mControlState = CPUT_CONTROL_ACTIVE;
                mSliderState = CPUT_SLIDER_NIB_UNPRESSED;
                handledCode = CPUT_EVENT_UNHANDLED;
                mbStartedClickInside = false;
                mbMouseInside = false;
            }
        }


    }
    else
    {
        mbMouseInside = false;
        mbStartedClickInside = false;
    }

    return handledCode;
}

// Calculate important size info used by most of the system
//--------------------------------------------------------------------------------
void CPUTSlider::CalculateLocationGuides(LocationGuides& guides)
{
    CPUT_RECT rect;
    mpControlText->GetDimensions(rect.width, rect.height);
    guides.TickIndent = mpActiveImageSizeList[0].width / 2.0f;  // cgrip
    guides.TextDownIndent = (float)(rect.height+2);
    guides.GripDownIndent = guides.TextDownIndent + mpActiveImageSizeList[1].height/2.0f; // CTick
    guides.TrayDownIndent =  guides.TextDownIndent +  (mpActiveImageSizeList[0].height / 2.0f) + mpActiveImageSizeList[1].height/2.0f; // CGrip CTick
    guides.TickSpacing = ( CPUT_DEFAULT_TRAY_WIDTH/(float)(mSliderNumberOfTicks-1));
    guides.StepSpacing = ( CPUT_DEFAULT_TRAY_WIDTH/(float)(mSliderNumberOfSteps-1));
    guides.TotalWidth = guides.TickIndent + mpActiveImageSizeList[2].width + CPUT_DEFAULT_TRAY_WIDTH + mpActiveImageSizeList[4].width; // CTrayLeftCap CTrayRightCap
    guides.TotalHeight = ( guides.TextDownIndent + guides.GripDownIndent + mpActiveImageSizeList[0].height); // CGrip
}


// return screen-space width/height of the control
//--------------------------------------------------------------------------------
void CPUTSlider::GetDimensions(int &width, int &height)
{
    LocationGuides guides;
    CalculateLocationGuides(guides);

    width=(int)guides.TotalWidth;
    height=(int)guides.TotalHeight;
}

// return the x/y window position of the control
//--------------------------------------------------------------------------------
void CPUTSlider::GetPosition(int &x, int &y)
{
    x = mControlDimensions.x;
    y = mControlDimensions.y;
}

//
//--------------------------------------------------------------------------------
bool CPUTSlider::ContainsPoint(int x, int y)
{
    // calculate all the locations we'll need for drawing
    LocationGuides guides;
    CalculateLocationGuides(guides);

    if( (x > ( mControlDimensions.x + guides.TotalWidth )) ||
        (x < mControlDimensions.x) ||
        (y < mControlDimensions.y) ||
        (y > (mControlDimensions.y + guides.TotalHeight)) )
    {
        return false;
    }

    return true;
}

//
//--------------------------------------------------------------------------------
bool CPUTSlider::PointingAtNub(int x, int y)
{
    // calculate all the locations we'll need for drawing
    LocationGuides guides;
    CalculateLocationGuides(guides);

    // locate the grabber coordinates
    float UpperLeftX=mSliderNubLocation+mControlDimensions.x;
    float UpperLeftY=(float)(mControlDimensions.y + guides.GripDownIndent);
    float LowerRightX = UpperLeftX + mpActiveImageSizeList[Grip].width;
    float LowerRightY = UpperLeftY + mpActiveImageSizeList[Grip].height;

    if( (x>LowerRightX) || (x<UpperLeftX) ||
        (y<UpperLeftY) || (y>LowerRightY ) )
    {
        return false;
    }

    return true;
}

//
//--------------------------------------------------------------------------------
void CPUTSlider::SnapToNearestTick()
{
    LocationGuides guides;
    CalculateLocationGuides(guides);


    float locationOnTray = mSliderNubLocation;

    int index = (int) (locationOnTray/guides.StepSpacing);
    float remainder = locationOnTray - index*guides.StepSpacing;
    if(remainder > (guides.StepSpacing/2.0f))
    {
        if(index<(mSliderNumberOfSteps-1))
        {
            // snap to next higher spot
            index++;
        }
        
        // on really large scales (tiny step sizes)
        // you can actually get over by a pixel 
        if(index>mSliderNumberOfSteps)
        {
            index = mSliderNumberOfSteps;
        }
    }

    // calculate the tick mark to align with
    mSliderNubLocation = (float)(guides.TickIndent  + (index*guides.StepSpacing) - (mpActiveImageSizeList[Grip].width/2.0f));
   
    Recalculate();

}

// Once we get above a certain number of ticks - the ticks overlap and are 
// useless, look jenky, and 
//------------------------------------------------------------------------------
void CPUTSlider::ClampTicks()
{
    // This number is somewhat arbitrary, but visually pleasing based on current
    // default GUI graphics.  If you change the graphics, you might want to
    // change this number based on size of the tick, default slider size, etc
    if(mSliderNumberOfTicks > 100)
    {
        mSliderNumberOfTicks = 100;
    }    
}

// Set the string that sits above the actual slider bar
//------------------------------------------------------------------------------
void CPUTSlider::SetText(const cString ControlText)
{
    mpControlText->SetText(ControlText);

    Recalculate();

    // position or size may move - force a recalculation of this control's location
    // if it is managed by the auto-arrange function
    if(this->IsAutoArranged())
    {
        mControlNeedsArrangmentResizing = true;
    }
    else
    {
        mControlGraphicsDirty = true;
    }
}

// Enable/disable the control
//--------------------------------------------------------------------------------
void CPUTSlider::SetEnable(bool in_bEnabled)
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
    mpControlText->SetEnable(in_bEnabled);

    // recalculate control's quads 
    Recalculate();
}

// Enable the drawing of the ticks
//------------------------------------------------------------------------------
void CPUTSlider::SetTickDrawing(bool DrawTicks)
{
    mbDrawTicks = DrawTicks;
    Recalculate();
}

//CPUTSlider

// Set the Scale
//--------------------------------------------------------------------------------
CPUTResult CPUTSlider::SetScale(float StartValue, float EndValue, int NumberOfSteps)
{
    ASSERT( StartValue < EndValue, _L("Slider start greater or equal to slider end") );
    ASSERT( NumberOfSteps > 1 , _L("Slider must have more than 2 steps from start to end value") );

    mSliderStartValue = StartValue;
    mSliderEndValue = EndValue;
    mSliderNumberOfSteps = NumberOfSteps;
    mSliderNumberOfTicks = mSliderNumberOfSteps;

    // re-sizes the mirror buffers to accomidate the new items
    RegisterInstanceResources();

    // to avoid the problem of changing the scale and having the gripper be
    // out of that range, setScale always sets the gripper to the start
    // value when re-ranging the control
    SetValue(StartValue);

    // Did we calculate a reasonable number of ticks?  Or should we clamp them to a 
    // visibly pleasing amount?
    ClampTicks();

    // we've likely moved things, so re-calculate the vertex buffer
    Recalculate();

    return CPUT_SUCCESS;
}

//--------------------------------------------------------------------------------
CPUTResult CPUTSlider::SetNumberOfTicks(int NumberOfTicks)
{
    // you cannot have more ticks than there are steps
    if(mSliderNumberOfSteps < NumberOfTicks)
    {
        return CPUT_ERROR_INVALID_PARAMETER;
    }

    mSliderNumberOfTicks = NumberOfTicks;

    // Did we calculate a reasonable number of ticks?  Or should we clamp them to a 
    // visibly pleasing amount?
    ClampTicks();

    // sometimes indent changes due to resetting, 
    // so recalculate
    SnapToNearestTick();

    return CPUT_SUCCESS;
}

// Get's the value the slider is current positioned at
//--------------------------------------------------------------------------------
CPUTResult CPUTSlider::GetValue(float & fValue)
{
    LocationGuides guides;
    CalculateLocationGuides(guides);
  
    float fractpart, intpart;
    fractpart = modf((mSliderNubLocation/guides.StepSpacing) , &intpart);
    int index = (int) intpart;
    if(fractpart>=0.5f)
        index++;

    float stepSizeValue = (mSliderEndValue - mSliderStartValue)/(float)(mSliderNumberOfSteps-1);

    // calculate the slider location's value
    fValue = mSliderStartValue + stepSizeValue*index;

    // In extreme range cases, the calculated value might be off by a fractional amount
    // prevent the slider from returning an above/below start/end value
    if(fValue > mSliderEndValue)
    {
        fValue = mSliderEndValue;
    }
    if(fValue < mSliderStartValue)
    {
        fValue = mSliderStartValue;
    }

    return CPUT_SUCCESS;
}

// Moves the slider gripper to the specified value on the slider
//--------------------------------------------------------------------------------
CPUTResult CPUTSlider::SetValue(float fValue)
{
    if(fValue>mSliderEndValue)
    {
        fValue = mSliderEndValue;
    }
    else if(fValue<mSliderStartValue)
    {
        fValue = mSliderStartValue;
    }

    LocationGuides guides;
    CalculateLocationGuides(guides);

    //+ guides.TickIndent ) + (i*guides.TickSpacing);
    float percentValue = (fValue-mSliderStartValue)/(mSliderEndValue-mSliderStartValue);
    float locationOnTray = guides.TotalWidth  *percentValue;

    int index = (int) (locationOnTray/guides.StepSpacing);
    float remainder = locationOnTray - index*guides.StepSpacing;
    if(remainder > (guides.StepSpacing/2.0f))
    {
        // snap to next higher spot
        index++;
    }

    float stepSizeValue = (mSliderEndValue - mSliderStartValue)/(mSliderNumberOfSteps-1);
    float remainderValue = (fValue-mSliderStartValue) - index*stepSizeValue;

    float remainderPercent = remainderValue/stepSizeValue;
    float remainderPixels = remainderPercent*guides.StepSpacing;
    mSliderNubLocation = (float)(guides.TickIndent  + (index*guides.StepSpacing) + remainderPixels - (mpActiveImageSizeList[Grip].width/2.0f));

    Recalculate();

    return CPUT_SUCCESS;
}

// 'Draw' this control into the supplied vertex buffer object
//--------------------------------------------------------------------------------
void CPUTSlider::DrawIntoBuffer(CPUTGUIVertex *pVertexBufferMirror, UINT *pInsertIndex, UINT pMaxBufferSize, CPUTGUIVertex *pTextVertexBufferMirror, UINT *pTextInsertIndex, UINT MaxTextVertexBufferSize)
{
    // don't bother drawing if control is invisible, bad buffer pointers, etc
    if(!mControlVisible)
    {
        return;
    }
    if((NULL==pVertexBufferMirror) || (NULL==pInsertIndex))
    {
        return;
    }
    if(!mpMirrorBufferActive || !mpMirrorBufferPressed)
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
        if(CPUT_SLIDER_NIB_PRESSED == mSliderState)
        {
            memcpy(&pVertexBufferMirror[*pInsertIndex], mpMirrorBufferPressed, sizeof(CPUTGUIVertex)*VertexCopyCount);
        }        
        else
        {
            memcpy(&pVertexBufferMirror[*pInsertIndex], mpMirrorBufferActive, sizeof(CPUTGUIVertex)*VertexCopyCount);
        }
        
        break;
    case CPUT_CONTROL_INACTIVE:
        // copy the inactive button into the stream
        memcpy(&pVertexBufferMirror[*pInsertIndex], mpMirrorBufferDisabled, sizeof(CPUTGUIVertex)*VertexCopyCount);
        break;

    default:
        // error! unknown state
        ASSERT(0,_L("CPUTCheckbox: Control is in unknown state"));
        return;
    }

    // move the uber-buffer index the tail of what we just added
    // each new quad has 6 verts in it (and each vert has 3+2 floats in it).
    *pInsertIndex+=VertexCopyCount;


    // now do the text
    // draw the text
    if(mpControlText)
    {
        mpControlText->DrawIntoBuffer((CPUTGUIVertex*) pTextVertexBufferMirror, pTextInsertIndex, MaxTextVertexBufferSize);
    }

    // we'll mark the control as no longer being 'dirty'
    mControlGraphicsDirty = false;
}

// This function re-calculates the positions of the various items in the control
//------------------------------------------------------------------------
void CPUTSlider::Recalculate()
{
    LocationGuides guides;
    CalculateLocationGuides(guides);

    // left tray cap
    float TrayX=(float)(mControlDimensions.x + guides.TickIndent/2.0f);
    float TrayY=(float)(mControlDimensions.y + guides.TrayDownIndent);
    AddQuadIntoMirrorBuffer(mpMirrorBufferActive,
                            0*6,
                            (float) TrayX, (float) TrayY,
                            (float) mpActiveImageSizeList[TrayLeftCap].width, (float) mpActiveImageSizeList[TrayLeftCap].height, 
                            mpUVCoordsSlider_active[TrayLeftCap*2+0], mpUVCoordsSlider_active[TrayLeftCap*2+1]
                        );
    AddQuadIntoMirrorBuffer(mpMirrorBufferPressed,
                            0*6,
                            (float) TrayX, (float) TrayY,
                            (float) mpActiveImageSizeList[TrayLeftCap].width, (float) mpActiveImageSizeList[TrayLeftCap].height, 
                            mpUVCoordsSlider_active[TrayLeftCap*2+0], mpUVCoordsSlider_active[TrayLeftCap*2+1]
                        );
    AddQuadIntoMirrorBuffer(mpMirrorBufferDisabled,
                            0*6,
                            (float) TrayX, (float) TrayY,
                            (float) mpDisabledImageSizeList[TrayLeftCap].width, (float) mpDisabledImageSizeList[TrayLeftCap].height, 
                            mpUVCoordsSlider_disabled[TrayLeftCap*2+0], mpUVCoordsSlider_disabled[TrayLeftCap*2+1]
                        );

    // tray
    TrayX=(float)(mControlDimensions.x + guides.TickIndent/2.0f + mpActiveImageSizeList[2].width); // +tray cap left
    TrayY=(float)(mControlDimensions.y + guides.TrayDownIndent);
    AddQuadIntoMirrorBuffer(mpMirrorBufferActive,
                            1*6,
                            (float) TrayX, (float) TrayY,
                            //(float) mpActiveImageSizeList[3].width+50,
                            (CPUT_DEFAULT_TRAY_WIDTH+guides.TickIndent/2.0f), (float) mpActiveImageSizeList[Tray].height, 
                            mpUVCoordsSlider_active[Tray*2+0], mpUVCoordsSlider_active[Tray*2+1]
                        );
    AddQuadIntoMirrorBuffer(mpMirrorBufferPressed,
                            1*6,
                            (float) TrayX, (float) TrayY,
                            //(float) mpActiveImageSizeList[3].width+50,
                            (CPUT_DEFAULT_TRAY_WIDTH+guides.TickIndent/2.0f), (float) mpActiveImageSizeList[Tray].height, 
                            mpUVCoordsSlider_active[Tray*2+0], mpUVCoordsSlider_active[Tray*2+1]
                        );  
    AddQuadIntoMirrorBuffer(mpMirrorBufferDisabled,
                            1*6,
                            (float) TrayX, (float) TrayY,
                            //(float) mpActiveImageSizeList[3].width+50,
                            (CPUT_DEFAULT_TRAY_WIDTH+guides.TickIndent/2.0f), (float) mpDisabledImageSizeList[Tray].height, 
                            mpUVCoordsSlider_disabled[Tray*2+0], mpUVCoordsSlider_disabled[Tray*2+1]
                        );  

    // right tray cap
    TrayX=(float)(mControlDimensions.x  + guides.TickIndent/2.0f + mpActiveImageSizeList[3].width + CPUT_DEFAULT_TRAY_WIDTH + guides.TickIndent/2.0f + 1.0);
    TrayY=(float)(mControlDimensions.y + guides.TrayDownIndent);
    AddQuadIntoMirrorBuffer(mpMirrorBufferActive,
                            2*6,
                            (float) TrayX, (float) TrayY,
                            (float) mpActiveImageSizeList[TrayRightCap].width, (float) mpActiveImageSizeList[TrayRightCap].height, 
                            mpUVCoordsSlider_active[TrayRightCap*2+0], mpUVCoordsSlider_active[TrayRightCap*2+1]
                        );
    AddQuadIntoMirrorBuffer(mpMirrorBufferPressed,
                            2*6,
                            (float) TrayX, (float) TrayY,
                            (float) mpActiveImageSizeList[TrayRightCap].width, (float) mpActiveImageSizeList[TrayRightCap].height, 
                            mpUVCoordsSlider_active[TrayRightCap*2+0], mpUVCoordsSlider_active[TrayRightCap*2+1]
                        );
    AddQuadIntoMirrorBuffer(mpMirrorBufferDisabled,
                            2*6,
                            (float) TrayX, (float) TrayY,
                            (float) mpDisabledImageSizeList[TrayRightCap].width, (float) mpDisabledImageSizeList[TrayRightCap].height, 
                            mpUVCoordsSlider_disabled[TrayRightCap*2+0], mpUVCoordsSlider_disabled[TrayRightCap*2+1]
                        );
    
    // nub
    TrayX = mControlDimensions.x + mSliderNubLocation;
    TrayY=(float)(mControlDimensions.y + guides.GripDownIndent);
    AddQuadIntoMirrorBuffer(mpMirrorBufferActive,
                            3*6,
                            (float) TrayX, (float) TrayY,
                            (float) mpActiveImageSizeList[Grip].width, (float) mpActiveImageSizeList[Grip].height, 
                            mpUVCoordsSlider_active[Grip*2+0], mpUVCoordsSlider_active[Grip*2+1]
                        );
    AddQuadIntoMirrorBuffer(mpMirrorBufferPressed,
                            3*6,
                            (float) TrayX, (float) TrayY,
                            (float) mpPressedImageSizeList[Grip].width, (float) mpPressedImageSizeList[Grip].height, 
                            mpUVCoordsSlider_pressed[Grip*2+0], mpUVCoordsSlider_pressed[Grip*2+1]
                        );
    AddQuadIntoMirrorBuffer(mpMirrorBufferDisabled,
                            3*6,
                            (float) TrayX, (float) TrayY,
                            (float) mpDisabledImageSizeList[Grip].width, (float) mpDisabledImageSizeList[Grip].height, 
                            mpUVCoordsSlider_disabled[Grip*2+0], mpUVCoordsSlider_disabled[Grip*2+1]
                        );


    if(mbDrawTicks)
    {
        // ticks
        int uberBufferIndex=4;
        for(int i=0; i<mSliderNumberOfTicks; i++)
        {
            TrayX=(float)(mControlDimensions.x  + guides.TickIndent ) + (i*guides.TickSpacing);
            TrayY=(float)(mControlDimensions.y + guides.TextDownIndent);
            AddQuadIntoMirrorBuffer(mpMirrorBufferActive,
                uberBufferIndex*6,
                (float) TrayX, (float) TrayY,
                (float) mpActiveImageSizeList[Tick].width, (float) mpActiveImageSizeList[Tick].height, 
                mpUVCoordsSlider_active[Tick*2+0], mpUVCoordsSlider_active[Tick*2+1]
            );
            AddQuadIntoMirrorBuffer(mpMirrorBufferPressed,
                uberBufferIndex*6,
                (float) TrayX, (float) TrayY,
                (float) mpActiveImageSizeList[Tick].width, (float) mpActiveImageSizeList[Tick].height, 
                mpUVCoordsSlider_active[Tick*2+0], mpUVCoordsSlider_active[Tick*2+1]
            );
            AddQuadIntoMirrorBuffer(mpMirrorBufferDisabled,
                uberBufferIndex*6,
                (float) TrayX, (float) TrayY,
                (float) mpDisabledImageSizeList[Tick].width, (float) mpDisabledImageSizeList[Tick].height, 
                mpUVCoordsSlider_disabled[Tick*2+0], mpUVCoordsSlider_disabled[Tick*2+1]
            );
            uberBufferIndex++;
        }
    }

    // control graphics are dirty
    mControlGraphicsDirty = true;
}



// This generates a quad with the supplied coordinates/uv's/etc.
//------------------------------------------------------------------------
void CPUTSlider::AddQuadIntoMirrorBuffer(CPUTGUIVertex *pMirrorBuffer, 
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



