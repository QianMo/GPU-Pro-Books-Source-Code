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
#include "CPUTGuiControllerDX11.h"
#include "CPUT_DX11.h" // for CPUTSetRasterizerState()
#include "CPUTTextureDX11.h"
#include "CPUTFontDX11.h"


CPUTGuiControllerDX11* CPUTGuiControllerDX11::mguiController = NULL;

// chained constructor
//--------------------------------------------------------------------------------
CPUTGuiControllerDX11::CPUTGuiControllerDX11():CPUTGuiController(),
    mpGUIVertexShader(NULL),
    mpGUIPixelShader(NULL),
    mpConstantBufferVS(NULL),
    mpVertexLayout(NULL),

    // texture atlas+uber buffer
    mpControlTextureAtlas(NULL),
    mpControlTextureAtlasView(NULL),
    mpUberBuffer(NULL),
    mpMirrorBuffer(NULL),
    mUberBufferIndex(0),
    mUberBufferMax(CPUT_GUI_BUFFER_SIZE),
    mpFont(NULL),
    mpTextTextureAtlas(NULL),
    mpTextTextureAtlasView(NULL),
    mpTextUberBuffer(NULL),
    mpTextMirrorBuffer(NULL),
    mTextUberBufferIndex(0),

    mFocusedControlBufferIndex(0),
    mpFocusedControlBuffer(NULL),
    mFocusedControlTextBufferIndex(0),
    mpFocusedControlTextBuffer(NULL),
    mpGUIRenderStateBlock(NULL),
    
    mbDrawFPS(false),
    mLastFPS(0),
    mLastFPSTime(0),
    mNumFramesRenderd(0),
    mFPSUpdatePeriod(0.5f),
    mpFPSCounter(NULL),
    mpFPSMirrorBuffer(NULL),
    mFPSBufferIndex(0),
    mpFPSDirectXBuffer(NULL),
    mpFPSTimer(NULL)
{
    mpMirrorBuffer = new CPUTGUIVertex[CPUT_GUI_BUFFER_SIZE];
    mpTextMirrorBuffer = new  CPUTGUIVertex[CPUT_GUI_BUFFER_STRING_SIZE];

    mpFocusedControlMirrorBuffer = new CPUTGUIVertex[CPUT_GUI_BUFFER_SIZE];
    mpFocusedControlTextMirrorBuffer = new CPUTGUIVertex[CPUT_GUI_BUFFER_STRING_SIZE];
    
    mpFPSMirrorBuffer = new CPUTGUIVertex[CPUT_GUI_BUFFER_STRING_SIZE];
    mpFPSTimer = new CPUTTimerWin();
    
#ifdef SAVE_RESTORE_DSHSGS_SHADER_STATE
    mpGeometryShaderState=NULL;
    mpGeometryShaderClassInstances=NULL;
    mGeometryShaderNumClassInstances=0;

    mpHullShaderState=NULL;
    mpHullShaderClassInstances=NULL;
    UINT mHullShaderNumClassInstance=0;

    mpDomainShaderState=NULL;
    mpDomainShaderClassIntances=NULL;
    UINT mDomainShaderNumClassInstances=0;
#endif
}

// destructor
//--------------------------------------------------------------------------------
CPUTGuiControllerDX11::~CPUTGuiControllerDX11()
{
    // delete all the controls under you
	ReleaseResources();
    DeleteAllControls();

    // FPS counter
    SAFE_DELETE(mpFPSCounter);
    SAFE_DELETE(mpFPSTimer);

    // delete arrays
    SAFE_DELETE_ARRAY(mpTextMirrorBuffer);
    SAFE_DELETE_ARRAY(mpMirrorBuffer);
    SAFE_DELETE_ARRAY(mpFocusedControlMirrorBuffer);
    SAFE_DELETE_ARRAY(mpFocusedControlTextMirrorBuffer);
    SAFE_DELETE_ARRAY(mpFPSMirrorBuffer);
}

// static getter
//--------------------------------------------------------------------------------
CPUTGuiControllerDX11* CPUTGuiControllerDX11::GetController()
{
    if(NULL==mguiController)
        mguiController = new CPUTGuiControllerDX11();
    return mguiController;
}


// Delete the controller
//--------------------------------------------------------------------------------
CPUTResult CPUTGuiControllerDX11::DeleteController()
{
    SAFE_DELETE(mguiController);

    return CPUT_SUCCESS;
}

//--------------------------------------------------------------------------------
CPUTResult CPUTGuiControllerDX11::ReleaseResources()
{
	//Release all allocated resources
    SAFE_RELEASE(mpGUIVertexShader);
    SAFE_RELEASE(mpGUIPixelShader);
    SAFE_RELEASE(mpVertexLayout);
    SAFE_RELEASE(mpConstantBufferVS);

    // release the texture atlas+buffers
    SAFE_RELEASE(mpControlTextureAtlasView);
    SAFE_RELEASE(mpControlTextureAtlas);
    SAFE_RELEASE(mpUberBuffer);
    
//	SAFE_RELEASE(mpFont->mpTextAtlas);
    SAFE_RELEASE(mpFont);
    SAFE_RELEASE(mpTextTextureAtlasView);
    SAFE_RELEASE(mpTextTextureAtlas);    
    SAFE_RELEASE(mpTextUberBuffer);
    
    SAFE_RELEASE(mpFocusedControlBuffer);
    SAFE_RELEASE(mpFocusedControlTextBuffer);    

    SAFE_RELEASE(mpGUIRenderStateBlock);

    SAFE_RELEASE(mpFPSDirectXBuffer);

    // tell all controls to unregister all their static resources
    CPUTText::UnRegisterStaticResources();
    CPUTButton::UnRegisterStaticResources();
    CPUTCheckbox::UnRegisterStaticResources();
    CPUTSlider::UnRegisterStaticResources();
    CPUTDropdown::UnRegisterStaticResources();

	return CPUT_SUCCESS;
}

// Initialize the GUI controller and all it's static resources
//--------------------------------------------------------------------------------
CPUTResult CPUTGuiControllerDX11::Initialize(ID3D11DeviceContext *pImmediateContext, cString &ResourceDirectory)
{
    // All the GUI resource files
    cString VertexShader =      _L(".//shaders//GUIShaderDX.vs");
    cString PixelShader =       _L(".//shaders//GUIShaderDX.ps");
    cString DefaultFontFile =   _L(".//fonts//font_arial_12.dds");
    cString ControlAtlasTexture = _L(".//controls//atlas.dds");
    cString RenderStateFile =   _L(".//Shaders//GUIRenderState.rs");

    // store the resource directory to be used by the GUI manager
    SetResourceDirectory(ResourceDirectory);

    return RegisterGUIResources(pImmediateContext, VertexShader, PixelShader, RenderStateFile, DefaultFontFile, ControlAtlasTexture);
}


// Control creation

// Create a button control and add it to the GUI layout controller
//--------------------------------------------------------------------------------
CPUTResult CPUTGuiControllerDX11::CreateButton(const cString ButtonText, CPUTControlID controlID, CPUTControlID panelID, CPUTButton **ppButton)
{
    // create the control
    CPUTButton *pButton = new CPUTButton(ButtonText, controlID, mpFont);
    ASSERT(NULL != pButton, _L("Failed to create control.") );

    // return control if requested
    if(NULL!=ppButton)
    {
        *ppButton = pButton;
    }

    // add control to the gui manager
    return this->AddControl(pButton, panelID);

}

// Create a slider control and add it to the GUI layout controller
//--------------------------------------------------------------------------------
CPUTResult CPUTGuiControllerDX11::CreateSlider(const cString SliderText, CPUTControlID controlID, CPUTControlID panelID, CPUTSlider **ppSlider)
{
    // create the control
    CPUTSlider *pSlider = new CPUTSlider(SliderText, controlID, mpFont);
    ASSERT(NULL!=pSlider, _L("Failed creating slider") );

    // return control if requested
    if(NULL!=ppSlider)
    {
        *ppSlider = pSlider;
    }
    
    // add control to the gui manager
    return this->AddControl(pSlider, panelID);
}

// Create a checkbox control and add it to the GUI layout controller
//--------------------------------------------------------------------------------
CPUTResult CPUTGuiControllerDX11::CreateCheckbox(const cString CheckboxText, CPUTControlID controlID, CPUTControlID panelID, CPUTCheckbox **ppCheckbox)
{
    // create the control
    CPUTCheckbox *pCheckbox = new CPUTCheckbox(CheckboxText, controlID, mpFont);
    ASSERT(NULL!=pCheckbox, _L("Failed creating checkbox") );

    // return control if requested
    if(NULL!=ppCheckbox)
    {
        *ppCheckbox = pCheckbox;
    }

    // add control to the gui manager
    return this->AddControl(pCheckbox, panelID);
}

// Create a dropdown control and add it to the GUI layout controller
//--------------------------------------------------------------------------------
CPUTResult CPUTGuiControllerDX11::CreateDropdown(const cString SelectionText, CPUTControlID controlID, CPUTControlID panelID, CPUTDropdown **ppDropdown)
{
    // create the control
    CPUTDropdown *pDropdown = new CPUTDropdown(SelectionText, controlID, mpFont);
    ASSERT(NULL!=pDropdown, _L("Failed creating control") );

    // return control if requested
    if(NULL!=ppDropdown)
    {
        *ppDropdown = pDropdown;
    }

    // add control to the gui manager
    CPUTResult result;
    result = this->AddControl(pDropdown, panelID);

    return result;
}

// Create a text item (static text)
//--------------------------------------------------------------------------------
CPUTResult CPUTGuiControllerDX11::CreateText(const cString Text, CPUTControlID controlID, CPUTControlID panelID, CPUTText **ppStatic)
{
    // create the control
    CPUTText *pStatic=NULL;
    pStatic = new CPUTText(Text, controlID, mpFont);
    ASSERT(NULL!=pStatic, _L("Failed creating static") );
    if(NULL != ppStatic)
    {
        *ppStatic = pStatic;
    }

    // add control to the gui manager
    return this->AddControl(pStatic, panelID);
}

// Deletes a control from the GUI manager
// Will delete all instances of the control no matter which panel(s) it is in and then
// deallocates the memory for the control
//--------------------------------------------------------------------------------
CPUTResult CPUTGuiControllerDX11::DeleteControl(CPUTControlID controlID)
{
    // look thruogh all the panels and delete the item with this controlID
    // for each panel
    std::vector <CPUTControl*> pDeleteList;

    for(UINT i=0; i<mControlPanelIDList.size(); i++)
    {
        // walk list of controls
        for(UINT j=0; j<mControlPanelIDList[i]->mControlList.size(); j++)
        {
            if(controlID == mControlPanelIDList[i]->mControlList[j]->GetControlID())
            {
                // found an instance of the control we wish to delete
                // see if it's in the list already
                bool bFound = false;
                for(UINT x=0; x<pDeleteList.size(); x++)
                {
                    if( mControlPanelIDList[i]->mControlList[j] ==  pDeleteList[x] )
                    {
                        bFound = true;
                        break;
                    }
                }

                if(!bFound)
                {
                    // store for deleting
                    pDeleteList.push_back( mControlPanelIDList[i]->mControlList[j] );
                }

                // remove the control from the container list
                mControlPanelIDList[i]->mControlList.erase( mControlPanelIDList[i]->mControlList.begin() + j );
            }
        }
    }

    // delete the control(s) we found with this id
    for(UINT i=0; i<pDeleteList.size(); i++)
    {
        SAFE_DELETE( pDeleteList[i] );
    }

    // force a resize event to recalculate new control locations now that some might have been deleted
    this->Resize();

    // don't worry about cleaning up std::vector list itself, it'll do so when it falls out of scope
    return CPUT_SUCCESS;
}

// DrawFPS - Should the GUI draw the FPS counter in the upper-left?
//--------------------------------------------------------------------------------
void CPUTGuiControllerDX11::DrawFPS(bool drawfps)
{
    mbDrawFPS = drawfps;
}

// GetFPS - Returns the last frame's FPS value
//--------------------------------------------------------------------------------
float CPUTGuiControllerDX11::GetFPS()
{
    return mLastFPS;
}

// Draw - must be positioned after all the controls are defined
//--------------------------------------------------------------------------------
void CPUTGuiControllerDX11::Draw(ID3D11DeviceContext *pImmediateContext, int SyncInterval)
{
    HEAPCHECK;

    if( 0 != GetNumberOfControlsInPanel())
    {
        SetGUIDrawingState(pImmediateContext);
    }
    else
    {
        return;
    }

    ID3D11VertexShader *pVertexShader = mpGUIVertexShader->GetNativeVertexShader();
    ID3D11PixelShader  *pPixelShader  = mpGUIPixelShader->GetNativePixelShader();

    // check and see if any of the controls resized themselves
    int ControlCount=GetNumberOfControlsInPanel();
    bool ResizingNeeded = false;
    for(int ii=0; ii<ControlCount; ii++)
    {
        CPUTControl *pControl = mControlPanelIDList[mActiveControlPanelSlotID]->mControlList[ii];
        if(true == pControl->ControlResizedItself())
        {
            ResizingNeeded = true;
            pControl->ControlResizingHandled();
        }
    }

    // if any of the controls resized, then re-do the autoarrangment
    if(true == ResizingNeeded)
    {
        this->Resize();
    }

    // Now check to see if any controls' graphics are dirty
    for(int ii=0; ii<ControlCount; ii++)
    {
        CPUTControl *pControl = mControlPanelIDList[mActiveControlPanelSlotID]->mControlList[ii];
        if(true == pControl->ControlGraphicsDirty())
        {
            mUberBufferDirty = true;
            break;
        }
    }

    // if any of the controls have announced they are dirty, then rebuild the mirror buffer and update the GFX buffer
    if(mUberBufferDirty)
    {
        
        // if a resize was flagged, do it now.  
        if(mRecalculateLayout)
        {
            RecalculateLayout();
        }


        // 'clear' the buffer by resetting the pointer to the head
        mUberBufferIndex = 0;
        mTextUberBufferIndex = 0;
        mFocusedControlBufferIndex = 0;
        mFocusedControlTextBufferIndex = 0;

        int ii=0;
        while(ii<GetNumberOfControlsInPanel())
        {
            CPUTControl *pControl = mControlPanelIDList[mActiveControlPanelSlotID]->mControlList[ii];

            // don't draw the focus control - draw it last so it stays on 'top'
            if(mpFocusControl != pControl)
            {
                switch(pControl->GetType())
                {
                case CPUT_BUTTON:
                    ((CPUTButton*)pControl)->DrawIntoBuffer(mpMirrorBuffer, &mUberBufferIndex, mUberBufferMax, mpTextMirrorBuffer, &mTextUberBufferIndex, CPUT_GUI_BUFFER_STRING_SIZE);                    
                    break;
                case CPUT_CHECKBOX:
                    ((CPUTCheckbox*)pControl)->DrawIntoBuffer(mpMirrorBuffer, &mUberBufferIndex, mUberBufferMax, mpTextMirrorBuffer, &mTextUberBufferIndex, CPUT_GUI_BUFFER_STRING_SIZE);
                    break;
                case CPUT_SLIDER:
                    ((CPUTSlider*)pControl)->DrawIntoBuffer(mpMirrorBuffer, &mUberBufferIndex, mUberBufferMax, mpTextMirrorBuffer, &mTextUberBufferIndex, CPUT_GUI_BUFFER_STRING_SIZE);
                    break;
                case CPUT_DROPDOWN:
                    ((CPUTDropdown*)pControl)->DrawIntoBuffer(mpMirrorBuffer, &mUberBufferIndex, mUberBufferMax, mpTextMirrorBuffer, &mTextUberBufferIndex, CPUT_GUI_BUFFER_STRING_SIZE);
                    break;

                case CPUT_STATIC:
                    ((CPUTText*)pControl)->DrawIntoBuffer(mpTextMirrorBuffer, &mTextUberBufferIndex, CPUT_GUI_BUFFER_STRING_SIZE);
                    break;
                }
            }
            ii++;
            HEAPCHECK
        }

        // do the 'focused' control last so it stays on top (i.e. dropdowns)
        if(mpFocusControl)
        {
            switch(mpFocusControl->GetType())
            {
            case CPUT_BUTTON:
                ((CPUTButton*)mpFocusControl)->DrawIntoBuffer(mpFocusedControlMirrorBuffer, &mFocusedControlBufferIndex, mUberBufferMax, mpFocusedControlTextMirrorBuffer, &mFocusedControlTextBufferIndex, CPUT_GUI_BUFFER_STRING_SIZE);                    
                break;
            case CPUT_CHECKBOX:
                ((CPUTCheckbox*)mpFocusControl)->DrawIntoBuffer(mpFocusedControlMirrorBuffer, &mFocusedControlBufferIndex, mUberBufferMax, mpFocusedControlTextMirrorBuffer, &mFocusedControlTextBufferIndex, CPUT_GUI_BUFFER_STRING_SIZE);
                break;
            case CPUT_SLIDER:
                ((CPUTSlider*)mpFocusControl)->DrawIntoBuffer(mpFocusedControlMirrorBuffer, &mFocusedControlBufferIndex, mUberBufferMax, mpFocusedControlTextMirrorBuffer, &mFocusedControlTextBufferIndex, CPUT_GUI_BUFFER_STRING_SIZE);
                break;
            case CPUT_DROPDOWN:
                ((CPUTDropdown*)mpFocusControl)->DrawIntoBuffer(mpFocusedControlMirrorBuffer, &mFocusedControlBufferIndex, mUberBufferMax, mpFocusedControlTextMirrorBuffer, &mFocusedControlTextBufferIndex, CPUT_GUI_BUFFER_STRING_SIZE);
                break;
            case CPUT_STATIC:
                ((CPUTText*)mpFocusControl)->DrawIntoBuffer(mpFocusedControlMirrorBuffer, &mFocusedControlTextBufferIndex, CPUT_GUI_BUFFER_STRING_SIZE);
                break;
            }
        }
        
                
        // update the uber-buffers with the control graphics
        UpdateUberBuffers(pImmediateContext);

        // Clear dirty flag on uberbuffer
        mUberBufferDirty = false;

    }
    HEAPCHECK



    // calculate the fps
    double time = mpFPSTimer->GetTotalTime();
    ++mNumFramesRenderd;
    if(time - mLastFPSTime > mFPSUpdatePeriod )
    {
        mLastFPS = (float)(mNumFramesRenderd / (time - mLastFPSTime));
        mLastFPSTime = time;
        mNumFramesRenderd = 0;
    }

    // if we're drawing the FPS counter - update that
    // We do this independently of uber-buffer updates since we'll have FPS updates every frame,
    // but likely not have control updates every frame
    if(mbDrawFPS)
    {
        // calculate the time elapsed since last frame
        bool UberBufferWasDirty = mUberBufferDirty;

        cString Data;
        {
            wchar_t wcstring[CPUT_MAX_STRING_LENGTH];
            swprintf_s(&wcstring[0], CPUT_MAX_STRING_LENGTH, _L("%.0f"), mLastFPS);
            Data=wcstring;
        }
        // build the FPS string
        cString FPS = _L("FPS: ")+Data;
        if( SyncInterval )
            FPS += L" (VSync)";
        mpFPSCounter->SetText(FPS);        

        // 'draw' the string into the buffer
        mFPSBufferIndex = 0;
        mpFPSCounter->DrawIntoBuffer(mpFPSMirrorBuffer, &mFPSBufferIndex, CPUT_GUI_BUFFER_STRING_SIZE);

        // update the DirectX vertex buffer
        ASSERT(CPUT_GUI_BUFFER_STRING_SIZE > mFocusedControlTextBufferIndex, _L("CPUT GUI: Too many strings for default-sized uber-buffer.  Increase CPUT_GUI_BUFFER_STRING_SIZE"));
        pImmediateContext->UpdateSubresource(mpFPSDirectXBuffer, 0, NULL, mpFPSMirrorBuffer, mFPSBufferIndex*sizeof(CPUTGUIVertex), 0);
        
        // start next frame timer
        mpFPSTimer->StartTimer();
        if(false == UberBufferWasDirty)
        {
            mUberBufferDirty = false;
        }
    }


    // set up orthographic display
    int windowWidth, windowHeight;
    GUIConstantBufferVS ConstantBufferMatrices;
    float znear = 0.1f;
    float zfar = 100.0f;
    XMMATRIX m;

    CPUTOSServices *pServices = CPUTOSServices::GetOSServices();
    pServices->GetClientDimensions( &windowWidth, &windowHeight );
    m = XMMatrixOrthographicOffCenterLH(0, (float)windowWidth, (float)windowHeight, 0, znear, zfar);
    ConstantBufferMatrices.Projection = XMMatrixTranspose( m );

    // set the vertex shader
    pImmediateContext->VSSetShader( pVertexShader, NULL, 0 );
    UINT VertexStride = sizeof(CPUTGUIVertex);
    UINT VertexOffset = 0;
    pImmediateContext->IASetVertexBuffers( 0, 1, &mpUberBuffer, &VertexStride, &VertexOffset );

    m = XMMatrixIdentity();
    ConstantBufferMatrices.Model = XMMatrixTranspose( m );
    pImmediateContext->UpdateSubresource( mpConstantBufferVS, 0, NULL, &ConstantBufferMatrices, 0, 0 );
    pImmediateContext->VSSetConstantBuffers( 0, 1, &mpConstantBufferVS );

    // -- draw the normal controls --    
    // draw the control graphics
    pImmediateContext->PSSetShader( pPixelShader, NULL, 0 );
    pImmediateContext->PSSetShaderResources( 0, 1, &mpControlTextureAtlasView );    
    pImmediateContext->Draw(mUberBufferIndex,0);

    // draw the control's text
    pImmediateContext->PSSetShaderResources( 0, 1, &mpTextTextureAtlasView );
    pImmediateContext->IASetVertexBuffers( 0, 1, &mpTextUberBuffer, &VertexStride, &VertexOffset );
    // draw the text uber-buffer
    pImmediateContext->Draw(mTextUberBufferIndex,0);

    // draw the FPS counter
    if(mbDrawFPS)
    {
        pImmediateContext->IASetVertexBuffers( 0, 1, &mpFPSDirectXBuffer, &VertexStride, &VertexOffset );
        pImmediateContext->Draw(mFPSBufferIndex, 0);
    }
    
    // -- draw the focused control --
    // Draw the focused control's graphics
    pImmediateContext->PSSetShader( pPixelShader, NULL, 0 );
    pImmediateContext->PSSetShaderResources( 0, 1, &mpControlTextureAtlasView );
    pImmediateContext->IASetVertexBuffers( 0, 1, &mpFocusedControlBuffer, &VertexStride, &VertexOffset );
    // draw the uber-buffer
    pImmediateContext->Draw(mFocusedControlBufferIndex,0);


    // Draw the focused control's text
    pImmediateContext->PSSetShaderResources( 0, 1, &mpTextTextureAtlasView );
    pImmediateContext->IASetVertexBuffers( 0, 1, &mpFocusedControlTextBuffer, &VertexStride, &VertexOffset );
    // draw the text uber-buffer
    pImmediateContext->Draw(mFocusedControlTextBufferIndex,0);


    // restore the drawing state
    ClearGUIDrawingState(pImmediateContext);
    HEAPCHECK;
}

//
//------------------------------------------------------------------------
CPUTResult CPUTGuiControllerDX11::UpdateUberBuffers(ID3D11DeviceContext *pImmediateContext )
{
    ASSERT(pImmediateContext, _L("CPUTGuiControllerDX11::UpdateUberBuffers - Context pointer is NULL"));

    // get the device
    ID3D11Device *pD3dDevice = NULL;
    pImmediateContext->GetDevice(&pD3dDevice);
    

    // Update geometry to draw the control graphics
    ASSERT(CPUT_GUI_VERTEX_BUFFER_SIZE > mUberBufferIndex, _L("CPUT GUI: Too many controls for default-sized uber-buffer.  Increase CPUT_GUI_VERTEX_BUFFER_SIZE"));
    pImmediateContext->UpdateSubresource(mpUberBuffer, 0, NULL, (void*) mpMirrorBuffer, sizeof( CPUTGUIVertex )*(mUberBufferIndex+1), 0);

    // Update geometry to draw the controls' text
    ASSERT(CPUT_GUI_BUFFER_STRING_SIZE > mTextUberBufferIndex, _L("CPUT GUI: Too many strings for default-sized uber-buffer.  Increase CPUT_GUI_BUFFER_STRING_SIZE"));
    pImmediateContext->UpdateSubresource(mpTextUberBuffer, 0, NULL, (void*) mpTextMirrorBuffer, sizeof( CPUTGUIVertex )*(mTextUberBufferIndex+1), 0);
    
    // register the focused control's graphics
    ASSERT(CPUT_GUI_VERTEX_BUFFER_SIZE > mUberBufferIndex, _L("CPUT GUI: Too many controls for default-sized uber-buffer.  Increase CPUT_GUI_VERTEX_BUFFER_SIZE"));
    pImmediateContext->UpdateSubresource(mpFocusedControlBuffer, 0, NULL, (void*) mpFocusedControlMirrorBuffer, sizeof( CPUTGUIVertex )*(mFocusedControlBufferIndex+1), 0);

    //register the focused control's text
    ASSERT(CPUT_GUI_BUFFER_STRING_SIZE > mFocusedControlTextBufferIndex, _L("CPUT GUI: Too many strings for default-sized uber-buffer.  Increase CPUT_GUI_BUFFER_STRING_SIZE"));
    pImmediateContext->UpdateSubresource(mpFocusedControlTextBuffer, 0, NULL, (void*) mpFocusedControlTextMirrorBuffer, sizeof( CPUTGUIVertex )*(mFocusedControlTextBufferIndex+1), 0);

    // release the device pointer
    SAFE_RELEASE(pD3dDevice);    

    return CPUT_SUCCESS;

}

// Set the state for drawing the gui
//-----------------------------------------------------------------------------
void CPUTGuiControllerDX11::SetGUIDrawingState(ID3D11DeviceContext *pImmediateContext)
{
    // set the GUI shaders as active
    ID3D11VertexShader *pVertexShader = mpGUIVertexShader->GetNativeVertexShader();
    pImmediateContext->VSSetShader( pVertexShader, NULL, 0 );
    
    //D3D11_VIEWPORT viewport  = { 0.0f, 0.0f, (float)mWidth, (float)mHeight, 0.0f, 1.0f };
    //((CPUTRenderParametersDX*)&renderParams)->mpContext->RSSetViewports( 1, &viewport );
    
#ifdef SAVE_RESTORE_DS_HS_GS_SHADER_STATE
    pImmediateContext->GSGetShader(&mpGeometryShaderState, &mpGeometryShaderClassInstances, &mGeometryShaderNumClassInstances);
    pImmediateContext->HSGetShader(&mpHullShaderState, &mpHullShaderClassInstances, &mHullShaderNumClassInstance);
    pImmediateContext->DSGetShader(&mpDomainShaderState, &mpDomainShaderClassIntances, &mDomainShaderNumClassInstances);
#endif

    // set the geometry, hull, and domain shaders to null (in case the user had set them)
    // since the GUI system doesn't need them
    pImmediateContext->GSSetShader(NULL, NULL, 0);
    pImmediateContext->HSSetShader(NULL, NULL, 0);
    pImmediateContext->DSSetShader(NULL, NULL, 0);
    
    // set topology to triangle list
    pImmediateContext->IAGetPrimitiveTopology(&mTopology);
    pImmediateContext->IASetPrimitiveTopology(D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        
    // Use CPUTRenderStateBlock to set the render state for GUI drawing
    CPUTRenderParametersDX renderParams;
    renderParams.mpContext = pImmediateContext;
    mpGUIRenderStateBlock->SetRenderStates(renderParams);

    // update the constant buffer matrices
    // set for orthographic perspective
    GUIConstantBufferVS cb;
    cb.Model = XMMatrixIdentity();
    cb.Projection = XMMatrixIdentity();
    pImmediateContext->UpdateSubresource( mpConstantBufferVS, 0, NULL, &cb, 0, 0 );
    pImmediateContext->VSSetConstantBuffers( 0, 1, &mpConstantBufferVS );

    // set the input layout
    pImmediateContext->IASetInputLayout( mpVertexLayout );

    // set the pixel shader
    pImmediateContext->PSSetShader( mpGUIPixelShader->GetNativePixelShader(), NULL, 0 );

}

// Restores the previous state
//-----------------------------------------------------------------------------
void CPUTGuiControllerDX11::ClearGUIDrawingState(ID3D11DeviceContext *pImmediateContext)
{
    // restore drawing topology mode that was in use before entering the gui drawing state
    pImmediateContext->IASetPrimitiveTopology(mTopology);

#ifdef SAVE_RESTORE_DS_HS_GS_SHADER_STATE
    pImmediateContext->GSSetShader(mpGeometryShaderState, (ID3D11ClassInstance* const*)&mpGeometryShaderClassInstances, mGeometryShaderNumClassInstances);
    pImmediateContext->HSSetShader(mpHullShaderState, (ID3D11ClassInstance* const*)&mpHullShaderClassInstances, mHullShaderNumClassInstance);
    pImmediateContext->DSSetShader(mpDomainShaderState, (ID3D11ClassInstance* const*)&mpDomainShaderClassIntances, mDomainShaderNumClassInstances);
#endif

}


// Load and register all the resources needed by the GUI system
//-----------------------------------------------------------------------------
CPUTResult CPUTGuiControllerDX11::RegisterGUIResources(ID3D11DeviceContext *pImmediateContext, cString VertexShaderFilename, cString PixelShaderFilename, cString RenderStateFile, cString DefaultFontFilename, cString ControlAtlasTexture)
{
    if(NULL==pImmediateContext)
    {
        return CPUT_ERROR_INVALID_PARAMETER;
    }
    CPUTResult result;
    HRESULT hr;
    ID3D11Device *pD3dDevice = NULL;
    CPUTOSServices *pServices = NULL;
    CPUTAssetLibraryDX11 *pAssetLibrary = NULL;
    cString ErrorMessage;

    // Get the services/resource pointers we need
    pServices = CPUTOSServices::GetOSServices();
    pImmediateContext->GetDevice(&pD3dDevice);
    pAssetLibrary = (CPUTAssetLibraryDX11*)CPUTAssetLibraryDX11::GetAssetLibrary();

    // Get the resource directory
    cString ResourceDirectory;
    CPUTGuiControllerDX11::GetController()->GetResourceDirectory(ResourceDirectory);

    // 1. Load the renderstate configuration for the GUI system
    mpGUIRenderStateBlock = (CPUTRenderStateBlockDX11*) pAssetLibrary->GetRenderStateBlock(ResourceDirectory+RenderStateFile); 
    ASSERT(mpGUIRenderStateBlock, _L("Error loading the render state file (.rs) needed for the CPUT GUI system"));
    

    // 2. Store the shader path from AssetLibrary, change it to OUR resource directory
    cString OriginalAssetLibraryDirectory = pAssetLibrary->GetShaderDirectory();
    pAssetLibrary->SetShaderDirectoryName(ResourceDirectory);
    

    // 3. load the shaders for gui drawing
    // Load the GUI Vertex Shader
    cString FullPath, FinalPath;
    FullPath = mResourceDirectory + VertexShaderFilename;
    pServices->ResolveAbsolutePathAndFilename(FullPath, &FinalPath);
    result = pAssetLibrary->GetVertexShader(FinalPath, pD3dDevice, _L("VS"), _L("vs_4_0"), &mpGUIVertexShader, true);
    CPUTSetDebugName( mpGUIVertexShader->GetNativeVertexShader(), _L("GUIVertexShader"));
    if(CPUTFAILED(result))
    {
        ASSERT(CPUTSUCCESS(result), _L("Error loading the vertex shader needed for the CPUT GUI system."));
    }
    ID3DBlob *pVertexShaderBlob = mpGUIVertexShader->GetBlob();

    // Load the GUI Pixel Shader
    FullPath = mResourceDirectory + PixelShaderFilename;
    pServices->ResolveAbsolutePathAndFilename(FullPath, &FinalPath);
    result = pAssetLibrary->GetPixelShader(FinalPath, pD3dDevice, _L("PS"), _L("ps_4_0"), &mpGUIPixelShader, true);
    CPUTSetDebugName( mpGUIPixelShader->GetNativePixelShader(), _L("GUIPixelShader"));
    if(CPUTFAILED(result))
    {
        ASSERT(CPUTSUCCESS(result), _L("Error loading the pixel shader needed for the CPUT GUI system."));
    }

    // Restore the previous shader directory
    pAssetLibrary->SetShaderDirectoryName(OriginalAssetLibraryDirectory);
    

    // 4. Create the vertex layout description for all the GUI controls we'll draw
    // set vertex shader as active so we can configure it
    ID3D11VertexShader *pVertexShader = mpGUIVertexShader->GetNativeVertexShader();
    pImmediateContext->VSSetShader( pVertexShader, NULL, 0 );

    D3D11_INPUT_ELEMENT_DESC layout[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },	        
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 20, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
    UINT numElements = ARRAYSIZE( layout );

    // Create the input layout
    hr = pD3dDevice->CreateInputLayout( layout, numElements, pVertexShaderBlob->GetBufferPointer(), pVertexShaderBlob->GetBufferSize(), &mpVertexLayout );
    ASSERT( SUCCEEDED(hr), _L("Error creating CPUT GUI system input layout" ));
    CPUTSetDebugName( mpVertexLayout, _L("CPUT GUI InputLayout object"));
    

    // 5. create the vertex shader constant buffer pointers
    D3D11_BUFFER_DESC bd;
    ZeroMemory( &bd, sizeof(bd) );
    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.ByteWidth = sizeof(GUIConstantBufferVS);
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bd.CPUAccessFlags = 0;
    hr = pD3dDevice->CreateBuffer( &bd, NULL, &mpConstantBufferVS );
    ASSERT( SUCCEEDED(hr), _L("Error creating constant buffer VS" ));
    CPUTSetDebugName( mpConstantBufferVS, _L("GUI ConstantBuffer"));
      
    // Set the texture directory for loading the control texture atlas
    pAssetLibrary->SetTextureDirectoryName(ResourceDirectory);

    // load the control atlas
    mpControlTextureAtlas = (CPUTTextureDX11*) pAssetLibrary->GetTexture(ControlAtlasTexture); 
    if(NULL==mpControlTextureAtlas)
    {
        return CPUT_TEXTURE_LOAD_ERROR;
    }
    mpControlTextureAtlasView = mpControlTextureAtlas->GetShaderResourceView();
    mpControlTextureAtlasView->AddRef();

    // restore the asset library's texture directory    
    pAssetLibrary->SetTextureDirectoryName(OriginalAssetLibraryDirectory);
    

    // 6. Load the font atlas
    // store the existing asset library font directory
    OriginalAssetLibraryDirectory = pAssetLibrary->GetFontDirectory(); 

    // set font directory to the resource directory
    pAssetLibrary->SetFontDirectoryName(ResourceDirectory);
    mpFont = (CPUTFontDX11*) pAssetLibrary->GetFont(DefaultFontFilename); 
    if(NULL==mpFont)
    {
        return CPUT_TEXTURE_LOAD_ERROR;
    }
    mpTextTextureAtlas = mpFont->GetAtlasTexture();
    mpTextTextureAtlas->AddRef();
    mpTextTextureAtlasView = mpFont->GetAtlasTextureResourceView();
    mpTextTextureAtlasView->AddRef();    

    // restore the asset library's font directory    
    pAssetLibrary->SetTextureDirectoryName(OriginalAssetLibraryDirectory);
    
    
    // 7. Set up the DirectX uber-buffers that the controls draw into
    int maxSize = max(CPUT_GUI_BUFFER_STRING_SIZE, CPUT_GUI_BUFFER_SIZE);
    maxSize = max(maxSize, CPUT_GUI_VERTEX_BUFFER_SIZE);    
    maxSize *= sizeof( CPUTGUIVertex );
    char *pZeroedBuffer= new char[maxSize];
    memset(pZeroedBuffer, 0, maxSize);

    // set up buffer description
    ZeroMemory( &bd, sizeof(bd) );
    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.ByteWidth = sizeof( CPUTGUIVertex ) * CPUT_GUI_VERTEX_BUFFER_SIZE; //mUberBufferIndex;
    bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    bd.CPUAccessFlags = 0;
    // initialization data (all 0's for now)
    D3D11_SUBRESOURCE_DATA InitData;
    ZeroMemory( &InitData, sizeof(InitData) );
    InitData.pSysMem = mpMirrorBuffer;

    // mpUberBuffer
    SAFE_RELEASE(mpUberBuffer);
    hr = pD3dDevice->CreateBuffer( &bd, &InitData, &mpUberBuffer );
    ASSERT( !FAILED( hr ), _L("CPUT GUI FPS counter buffer creation failure"));
    CPUTSetDebugName(mpUberBuffer, _L("CPUT GUI: Control's main vertex buffer"));


    // mpTextUberBuffer
    bd.ByteWidth = sizeof( CPUTGUIVertex ) * CPUT_GUI_BUFFER_STRING_SIZE;
    hr = pD3dDevice->CreateBuffer( &bd, &InitData, &mpTextUberBuffer );
    ASSERT( !FAILED( hr ), _L("CPUT GUI FPS counter buffer creation failure"));
    CPUTSetDebugName(mpTextUberBuffer, _L("CPUT GUI: control text vertex buffer"));


    // mpFocusedControlBuffer
    bd.ByteWidth = sizeof( CPUTGUIVertex ) * CPUT_GUI_VERTEX_BUFFER_SIZE;
    hr = pD3dDevice->CreateBuffer( &bd, &InitData, &mpFocusedControlBuffer );
    ASSERT( !FAILED( hr ), _L("CPUT GUI FPS counter buffer creation failure"));
    CPUTSetDebugName(mpFocusedControlBuffer, _L("CPUT GUI: focused control images vertex buffer"));

    
    // mpFocusedControlTextBuffer
    bd.ByteWidth = sizeof( CPUTGUIVertex ) * CPUT_GUI_BUFFER_STRING_SIZE; //mFocusedControlTextBufferIndex;
    hr = pD3dDevice->CreateBuffer( &bd, &InitData, &mpFocusedControlTextBuffer );
    ASSERT( !FAILED( hr ), _L("CPUT GUI FPS counter buffer creation failure"));
    CPUTSetDebugName(mpFocusedControlTextBuffer, _L("CPUT GUI: focused control text vertex buffer"));

    // mpFPSDirectXBuffer
    bd.ByteWidth = sizeof( CPUTGUIVertex ) * CPUT_GUI_BUFFER_STRING_SIZE;
    hr = pD3dDevice->CreateBuffer( &bd, &InitData, &mpFPSDirectXBuffer );
    ASSERT( !FAILED( hr ), _L("CPUT GUI FPS counter buffer creation failure"));
    CPUTSetDebugName(mpFPSDirectXBuffer, _L("CPUT GUI: FPS display text"));
    
    // no longer need the device - release it.
    SAFE_RELEASE(pD3dDevice);
    SAFE_DELETE_ARRAY(pZeroedBuffer);    


    // 8. Register all GUI sub-resources
    // Walk all the controls/fonts and have them register all their required static resources
    // Returning errors if you couldn't find your resources
    result = CPUTText::RegisterStaticResources();
    if(CPUTFAILED(result))
    {
        return result;
    }
    result = CPUTButton::RegisterStaticResources();
    if(CPUTFAILED(result))
    {
        return result;
    }
    result = CPUTCheckbox::RegisterStaticResources();
    if(CPUTFAILED(result))
    {
        return result;
    }
    result = CPUTSlider::RegisterStaticResources();
    if(CPUTFAILED(result))
    {
        return result;
    }
    result = CPUTDropdown::RegisterStaticResources();
    if(CPUTFAILED(result))
    {
        return result;
    }

    // create the FPS CPUTText object for drawing FPS
    mpFPSCounter = new CPUTText(_L("FPS:"), ID_CPUT_GUI_FPS_COUNTER, mpFont);
    mpFPSCounter->SetAutoArranged(false);
    mpFPSCounter->SetPosition(0,0);

    // start the timer
    mpFPSTimer->StartTimer();

    // done
    return CPUT_SUCCESS;
}
