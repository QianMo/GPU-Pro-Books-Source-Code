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
#ifndef __CPUTGUICONTROLLERDX11_H__
#define __CPUTGUICONTROLLERDX11_H__

#include "CPUTGuiController.h"
#include "CPUTTimerWin.h"

#include "CPUTButton.h"
#include "CPUTText.h"
#include "CPUTCheckbox.h"
#include "CPUTSlider.h"
#include "CPUTDropdown.h"
#include "CPUTVertexShaderDX11.h"
#include "CPUTPixelShaderDX11.h"
#include "CPUTRenderStateBlockDX11.h"

//#define SAVE_RESTORE_DS_HS_GS_SHADER_STATE

// forward declarations
class CPUT_DX11;
class CPUTButton;
class CPUTSlider;
class CPUTCheckbox;
class CPUTDropdown;
class CPUTText;
class CPUTTextureDX11;
class CPUTFontDX11;

const unsigned int CPUT_GUI_BUFFER_SIZE = 5000;         // size (in number of verticies) for all GUI control graphics
const unsigned int CPUT_GUI_BUFFER_STRING_SIZE = 5000;  // size (in number of verticies) for all GUI string graphics
const unsigned int CPUT_GUI_VERTEX_BUFFER_SIZE = 5000;
const CPUTControlID ID_CPUT_GUI_FPS_COUNTER = 4000000201;        // pick very random number for FPS counter string ID

#include <d3d11.h>
#include <xnamath.h> // for xmmatrix/et al


// the GUI controller class that dispatches the rendering calls to all the buttons
//--------------------------------------------------------------------------------
class CPUTGuiControllerDX11:public CPUTGuiController
{ 
    struct GUIConstantBufferVS
    {
        XMMATRIX Projection;
        XMMATRIX Model;
    };


public:
    static CPUTGuiControllerDX11 *GetController();
    static CPUTResult DeleteController();

    // initialization
    CPUTResult Initialize(ID3D11DeviceContext *pImmediateContext, cString &ResourceDirectory);
	CPUTResult ReleaseResources();
	

    // Control creation/deletion 'helpers'
    CPUTResult CreateButton(const cString pButtonText, CPUTControlID controlID, CPUTControlID panelID, CPUTButton **ppButton=NULL);
    CPUTResult CreateSlider(const cString pSliderText, CPUTControlID controlID, CPUTControlID panelID, CPUTSlider **ppSlider=NULL);
    CPUTResult CreateCheckbox(const cString pCheckboxText, CPUTControlID controlID, CPUTControlID panelID, CPUTCheckbox **ppCheckbox=NULL);
    CPUTResult CreateDropdown(const cString pSelectionText, CPUTControlID controlID, CPUTControlID panelID, CPUTDropdown **ppDropdown=NULL);
    CPUTResult CreateText(const cString Text,  CPUTControlID controlID, CPUTControlID panelID, CPUTText **ppStatic=NULL);    
    CPUTResult DeleteControl(CPUTControlID controlID);

    // draw routines    
    void Draw(ID3D11DeviceContext *pImmediateContext, int SyncInterval = 0);
    void DrawFPS(bool drawfps);
    float GetFPS();

private:
    static CPUTGuiControllerDX11 *mguiController; // singleton object

    // DirectX state objects for GUI drawing
    CPUTVertexShaderDX11 *mpGUIVertexShader;
    CPUTPixelShaderDX11  *mpGUIPixelShader;
    ID3D11InputLayout    *mpVertexLayout;
    ID3D11Buffer         *mpConstantBufferVS;
    GUIConstantBufferVS   mModelViewMatrices;

    // Texture atlas
    CPUTTextureDX11            *mpControlTextureAtlas;
    ID3D11ShaderResourceView   *mpControlTextureAtlasView;
    ID3D11Buffer               *mpUberBuffer;
    CPUTGUIVertex              *mpMirrorBuffer;
    UINT                        mUberBufferIndex;
    UINT                        mUberBufferMax;
    
    // Font atlas
    CPUTFontDX11               *mpFont;
    CPUTTextureDX11            *mpTextTextureAtlas;
    ID3D11ShaderResourceView   *mpTextTextureAtlasView;
    ID3D11Buffer               *mpTextUberBuffer;
    CPUTGUIVertex              *mpTextMirrorBuffer;
    UINT                        mTextUberBufferIndex;

    // Focused control buffers
    CPUTGUIVertex              *mpFocusedControlMirrorBuffer;
    UINT                        mFocusedControlBufferIndex;
    ID3D11Buffer               *mpFocusedControlBuffer;
    CPUTGUIVertex              *mpFocusedControlTextMirrorBuffer;
    UINT                        mFocusedControlTextBufferIndex;
    ID3D11Buffer               *mpFocusedControlTextBuffer;



    // FPS
    bool                        mbDrawFPS;
    float                       mLastFPS;
    float                       mFPSUpdatePeriod;
    double                      mLastFPSTime;
    float                       mNumFramesRenderd;
    CPUTText                   *mpFPSCounter;
    // FPS control buffers
    CPUTGUIVertex              *mpFPSMirrorBuffer;
    UINT                        mFPSBufferIndex;
    ID3D11Buffer               *mpFPSDirectXBuffer;
    CPUTTimerWin               *mpFPSTimer;

    // render state
    CPUTRenderStateBlockDX11   *mpGUIRenderStateBlock;
    CPUTResult UpdateUberBuffers(ID3D11DeviceContext *pImmediateContext );

#ifdef SAVE_RESTORE_DS_HS_GS_SHADER_STATE
    ID3D11GeometryShader   *mpGeometryShaderState;
    ID3D11ClassInstance    *mpGeometryShaderClassInstances;
    UINT                    mGeometryShaderNumClassInstances;

    ID3D11HullShader       *mpHullShaderState;
    ID3D11ClassInstance    *mpHullShaderClassInstances;
    UINT                    mHullShaderNumClassInstance;

    ID3D11DomainShader     *mpDomainShaderState;
    ID3D11ClassInstance    *mpDomainShaderClassIntances;
    UINT                    mDomainShaderNumClassInstances;
#endif


    // members for saving render state before/after drawing gui
    D3D11_PRIMITIVE_TOPOLOGY    mTopology;

    // helper functions
    CPUTGuiControllerDX11();    // singleton
    ~CPUTGuiControllerDX11();
    CPUTResult RegisterGUIResources(ID3D11DeviceContext *pImmediateContext, cString VertexShaderFilename, cString RenderStateFile, cString PixelShaderFilename, cString DefaultFontFilename, cString ControlTextureAtlas);
    void SetGUIDrawingState(ID3D11DeviceContext *pImmediateContext);
    void ClearGUIDrawingState(ID3D11DeviceContext *pImmediateContext);
};




#endif // #ifndef __CPUTGUICONTROLLERDX11_H__
