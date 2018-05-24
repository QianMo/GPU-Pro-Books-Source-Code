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
#ifndef __CPUTGUICONTROLLER_H__
#define __CPUTGUICONTROLLER_H__



#include <stdio.h>
#include <stdlib.h> // for RAND_MAX
#include <vector>

#include "CPUTEventHandler.h"
#include "CPUTControl.h"
#include "CPUTOSServicesWin.h"

// definition of the vertex used in the GUI shader
struct CPUTGUIVertex
{
    float3 Pos;
    float2 UV;
    CPUTColor4 Color;
};

class CPUTGuiController:public CPUTEventHandler
{
public:
    CPUTGuiController();
    virtual ~CPUTGuiController();

    // get GUI controller base class
    static CPUTGuiController *GetController();

    //CPUTEventHandler members
    virtual CPUTEventHandledCode HandleKeyboardEvent(CPUTKey key) {UNREFERENCED_PARAMETER(key); return CPUT_EVENT_UNHANDLED;}
    virtual CPUTEventHandledCode HandleMouseEvent(int x, int y, int wheel, CPUTMouseState state);

    // members
    void EnableAutoLayout(bool UseAutoLayout);
    CPUTResult SetResourceDirectory(const cString ResourceDirectory);     // sets the resource directory to use when loading GUI resources
    void GetResourceDirectory(cString &ResourceDirectory);           // sets the resource directory to use when loading GUI resources

    // Panels
    CPUTResult      AddControl(CPUTControl *pControl, CPUTControlID panelID);                                // adds a control to the specified panel
    CPUTResult      FindControl(CPUTControlID controlID, CPUTControl **ppControl, CPUTControlID *pPanelID);  // search all panels to find a control and its panelID
    CPUTControl    *GetControl(CPUTControlID controlID, CPUTResult *pResult = NULL);            // search all panels to find a control
    CPUTResult      SetActivePanel(CPUTControlID panelID);                                      // sets the actively displayed panel
    CPUTControlID   GetActivePanelID();                                                         // returns the ID of the active panel
    CPUTResult      GetActiveControlList(std::vector<CPUTControl*> *ppControlList);
    CPUTResult      RemoveControlFromPanel(CPUTControlID controlID, CPUTControlID panelID=-1);  // removes specified control from the panel (does not delete the control)
    CPUTResult      DeletePanel(CPUTControlID panelID);                                         // removes panel and deletes all controls associated with it
    void            DeleteAllControls();                                                        // deletes all controls and all panels
    int             GetNumberOfControlsInPanel(CPUTControlID panelID=-1);                       // returns the number of controls in a specific panel
    bool            IsControlInPanel(CPUTControlID controlID, CPUTControlID panelID=-1);        // Is specified control in panel?
    bool            IsRecalculatingLayout() {return mRecalculateLayout;}

    // drawing/callbacks
    void Resize();  
    void RecalculateLayout();                                                   // forces a recalculation of control positions
    void SetCallback(CPUTCallbackHandler *pHandler, bool ForceAll=false);       // sets the event handler callback on all registered controls
    void ControlIsDirty() {mUberBufferDirty = true;}

protected:
    cString     mResourceDirectory;
    bool        mUberBufferDirty;
    bool        mRecalculateLayout;

    struct Panel
    {
        CPUTControlID mpanelID;
        CPUTControl *mpFocusControl;
        std::vector<CPUTControl*> mControlList;
    };

    // list of panels which have lists of controls associated with it
    std::vector<Panel*>  mControlPanelIDList;

    // the active panel list
    CPUTControlID  mActiveControlPanelSlotID;
    CPUTCallbackHandler *mpHandler;

    // active control
    CPUTControl *mpFocusControl;

    bool mbAutoLayout;

private:
    bool mbRebuildDrawList;
    // helper functions
    UINT FindPanelIDIndex(CPUTControlID panelID);

};



#endif //#ifndef __CPUTGUICONTROLLER_H__
