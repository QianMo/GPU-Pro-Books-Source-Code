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
#ifndef __CPUTDROPDOWN_H__
#define __CPUTDROPDOWN_H__

#include "CPUTControl.h"

#include <vector>


typedef enum CPUTDropdownGUIState
{
    CPUT_DROPDOWN_GUI_MOUSE_NEUTRAL,
    CPUT_DROPDOWN_GUI_MOUSE_PRESSED,
} CPUTDropdownGUIState;



// forward declarations
class CPUTTextureDX11;
struct CPUTGUIVertex;
class CPUTFont;
class CPUTText;

#define CPUT_NUM_IMAGES_IN_DROPDOWN_ARRAY 12
#define CPUT_NUM_QUADS_IN_DROPDOWN_ARRAY 18
#define CPUT_NUM_QUADS_IN_CLOSED_DROPDOWN 9
#define CPUT_DROPDOWN_TEXT_PADDING 2

const int CLeftTop = 0;
const int CLeftMid = 1;
const int CLeftBot = 2;
const int CMidTop = 3;
const int CMidMid = 4;
const int CMidBot = 5;
const int CRightTop = 6;
const int CRightMid = 7;
const int CRightBot = 8;
const int CButtonUp = 9;
const int CButtonDown = 10;

#define CPUT_NUM_IMAGES_IN_DROPDOWN  12


// Dropdown base - common functionality for all dropdown controls
//-----------------------------------------------------------------------------
class CPUTDropdown:public CPUTControl
{
public:
    // button should self-register with the GuiController on create
    CPUTDropdown(CPUTDropdown& copy);
    CPUTDropdown(const cString ControlName, CPUTControlID id, CPUTFont *pFont);
    virtual ~CPUTDropdown();

    // CPUTControl
    virtual void GetPosition(int &x, int &y);
    virtual unsigned int GetOutputVertexCount();

    //CPUTEventHandler
    virtual CPUTEventHandledCode HandleKeyboardEvent(CPUTKey key){UNREFERENCED_PARAMETER(key);return CPUT_EVENT_UNHANDLED;}  


    //CPUTEventHandler
    virtual CPUTEventHandledCode HandleMouseEvent(int x, int y, int wheel, CPUTMouseState state);

    //CPUTControl
    virtual void SetEnable(bool in_bEnabled);
    virtual bool IsEnabled();
    virtual bool ContainsPoint(int x, int y);
    virtual bool ContainsPointReadoutArea(int x, int y);
    virtual bool ContainsPointTrayArea(int x, int y);
    virtual void SetText(cString ControlText) {;} // doesn't apply to this control
    virtual void GetText(cString &ControlString) {UNREFERENCED_PARAMETER(ControlString);} // doesn't apply to this control
    virtual void GetDimensions(int &width, int &height);
    virtual void SetPosition(int x, int y);

    //CPUTDropdownDX11
    void NumberOfSelectableItems(UINT &count);
    void GetSelectedItem(UINT &index);
    void GetSelectedItem(cString &Item);
    void SetSelectedItem(const UINT index);
    CPUTResult AddSelectionItem(const cString Item, bool IsSelected=false);
    void DeleteSelectionItem(const UINT index);
    void DeleteSelectionItem(const cString string);
    
    // Draw
    void DrawIntoBuffer(CPUTGUIVertex *pVertexBufferMirror, UINT *pInsertIndex, UINT pMaxBufferSize, CPUTGUIVertex *pTextVertexBufferMirror, UINT *pTextInsertIndex, UINT MaxTextVertexBufferSize);

    // Register assets
    static CPUTResult RegisterStaticResources();
    static CPUTResult UnRegisterStaticResources();

    CPUTResult RegisterInstanceResources();
    CPUTResult UnRegisterInstanceResources();



protected:
    CPUT_RECT               mControlDimensions;
    CPUTGUIControlState     mControlState;
    CPUTDropdownGUIState    mControlGuiState;
    void InitialStateSet();



    // uber-buffer adds
    CPUTFont *mpFont;
    CPUTText *mpButtonText;
    static CPUT_SIZE mpDropdownIdleImageSizeList[CPUT_NUM_IMAGES_IN_DROPDOWN];    
    static CPUT_SIZE mpDropdownDisabledSizeList[CPUT_NUM_IMAGES_IN_DROPDOWN];

    // uber-buffer per-instance
    CPUT_SIZE mpDropdownIdleSizeList[CPUT_NUM_IMAGES_IN_DROPDOWN];    
    CPUT_SIZE mpDropdownDisabledList[CPUT_NUM_IMAGES_IN_DROPDOWN];

    CPUTGUIVertex *mpMirrorBufferActive;
    CPUTGUIVertex *mpMirrorBufferDisabled;
    void Recalculate();
    void AddQuadIntoMirrorBuffer(CPUTGUIVertex *pMirrorBuffer, int index, float x, float y, float w, float h, float3 uv1, float3 uv2 );


    // instance data
    
    CPUT_RECT mButtonRect;
    CPUT_RECT mReadoutRectInside;
    CPUT_RECT mReadoutRectOutside;

    CPUT_RECT mTrayDimensions;
    bool mbMouseInside;
    UINT mRevertItem;
    bool mbStartedClickInside;
    bool mbStartedClickInsideTray;

    // list of items in the dropdown
    UINT mSelectedItemIndex;            //zero-based index that tells what item is currently highlighted (soft selected)
    UINT mConfirmedSelectedItemIndex;   //zero-based index that tells what item was last confirm-selected (hard selected)
    std::vector<CPUTText*> mpListOfSelectableItems;
    CPUTText *mpSelectedItemCopy;

    bool mbSizeDirty;
    UINT mVertexStride; // stride and offsets of the image quads we're drawing on
    UINT mVertexOffset;

    // helper functions
    void CalculateButtonRect(CPUT_RECT& button);
    void CalculateReadoutRect(CPUT_RECT& inner, CPUT_RECT& outer);
    void CalculateTrayRect(CPUT_RECT& inner, CPUT_RECT& outer);

    void CalculateReadoutTextPosition(int &x, int &y);
    void CalculateMaxItemSize(int &width, int &height);

    // sets which item is to be highlighted on a dropdown list
    void SetHighlightedItem(const UINT index);

};

#endif //#ifndef __CPUTDROPDOWN_H__
