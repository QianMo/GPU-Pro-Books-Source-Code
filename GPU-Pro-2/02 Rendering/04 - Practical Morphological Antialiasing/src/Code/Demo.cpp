/*
 * Copyright (C) 2010 Jorge Jimenez (jim@unizar.es)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must display the name 'Jorge Jimenez' as
 *    'Real-Time Rendering R&D' in the credits of the application, if such
 *    credits exist. The author of this work must be notified via email
 *    (jim@unizar.es) in this case of redistribution.
 *
 * 3. Neither the name of copyright holders nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS
 * IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "DXUT.h"
#include "DXUTgui.h"
#include "DXUTsettingsDlg.h"
#include "SDKmisc.h"

#include <sstream>
#include <iomanip>

#include "Timer.h"
#include "RenderTarget.h"
#include "Copy.h"
#include "MLAA.h"

using namespace std;


const int HUD_WIDTH = 140;

CDXUTDialogResourceManager dialogResourceManager;
CD3DSettingsDlg settingsDialog;
CDXUTDialog hud;

Timer *timer = NULL;
ID3DX10Font *font = NULL;
ID3DX10Sprite *sprite = NULL;
CDXUTTextHelper *txtHelper = NULL;

RenderTarget *mainRenderTarget = NULL;
DepthStencil *depthStencil = NULL;
ID3D10Texture2D *backbufferTexture = NULL;
MLAA *mlaa = NULL;

ID3D10ShaderResourceView *testView = NULL;
ID3D10ShaderResourceView *testDepthView = NULL;

bool showHud = true;


#define IDC_TOGGLEFULLSCREEN      1
#define IDC_CHANGEDEVICE          2
#define IDC_LOADIMAGE             3
#define IDC_DETECTIONMODE         4
#define IDC_VIEWMODE              5
#define IDC_INPUT                 6
#define IDC_ANTIALIASING          7
#define IDC_PROFILE               8
#define IDC_MAXSEARCHSTEPS_LABEL  9
#define IDC_MAXSEARCHSTEPS       10
#define IDC_THRESHOLD_LABEL      11
#define IDC_THRESHOLD            12


bool CALLBACK isDeviceAcceptable(UINT adapter, UINT output, D3D10_DRIVER_TYPE deviceType, DXGI_FORMAT format, bool windowed, void *context) {
    return true;
}


void loadImage() {
    HRESULT hr;

    D3DX10_IMAGE_LOAD_INFO loadInfo = D3DX10_IMAGE_LOAD_INFO();
    loadInfo.MipLevels = 1;
    loadInfo.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    loadInfo.Filter = D3DX10_FILTER_POINT | D3DX10_FILTER_SRGB_IN;

    WCHAR *text = hud.GetComboBox(IDC_INPUT)->GetSelectedItem()->strText;
    if (int(hud.GetComboBox(IDC_INPUT)->GetSelectedData()) != -1) { // ... then, we have to search for it in the executable resources
        SAFE_RELEASE(testView);
        SAFE_RELEASE(testDepthView);

        // Load color
        V(D3DX10CreateShaderResourceViewFromResource(DXUTGetD3D10Device(), GetModuleHandle(NULL), text, &loadInfo, NULL, &testView, NULL));
                
        // Try to load depth
        loadInfo.Format = DXGI_FORMAT_R32_FLOAT;
        loadInfo.Filter = D3DX10_FILTER_POINT;

        wstring path = wstring(text).substr(0, wstring(text).find_last_of('.')) + L".dds";
        if (FindResource(GetModuleHandle(NULL), path.c_str(), RT_RCDATA) != NULL) {
            V(D3DX10CreateShaderResourceViewFromResource(DXUTGetD3D10Device(), GetModuleHandle(NULL), path.c_str(), &loadInfo, NULL, &testDepthView, NULL));
        }
    } else { // ... search for it in the file system
        ID3D10ShaderResourceView *view;
        if (FAILED(D3DX10CreateShaderResourceViewFromFile(DXUTGetD3D10Device(), text, &loadInfo, NULL, &view, NULL))) {
            MessageBox(NULL, L"Unable to open selected file", L"ERROR", MB_OK | MB_SETFOREGROUND | MB_TOPMOST);
        } else {
            SAFE_RELEASE(testView);
            SAFE_RELEASE(testDepthView);

            testView = view;

            ID3D10ShaderResourceView *depthView;
            wstring path = wstring(text).substr(0, wstring(text).find_last_of('.')) + L".dds";
            if (SUCCEEDED(D3DX10CreateShaderResourceViewFromFile(DXUTGetD3D10Device(), path.c_str(), &loadInfo, NULL, &depthView, NULL))) {
                testDepthView = depthView;
            }
        }
    }
    
    hud.GetComboBox(IDC_DETECTIONMODE)->SetEnabled(testDepthView != NULL);
    if (testDepthView == NULL) {
        hud.GetComboBox(IDC_DETECTIONMODE)->SetSelectedByIndex(0);
    }
}


void resizeWindow() {
    ID3D10Texture2D *texture2D;
    testView->GetResource(reinterpret_cast<ID3D10Resource **>(&texture2D));
    D3D10_TEXTURE2D_DESC desc;
    texture2D->GetDesc(&desc);
    texture2D->Release();

    RECT rect = {0, 0, desc.Width, desc.Height};
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
    SetWindowPos(DXUTGetHWNDDeviceWindowed(), 0, 0, 0,  (rect.right - rect.left), (rect.bottom - rect.top), SWP_NOZORDER | SWP_NOMOVE);
}


HRESULT CALLBACK onCreateDevice(ID3D10Device *device, const DXGI_SURFACE_DESC *desc, void *context) {
    HRESULT hr;
    V_RETURN(dialogResourceManager.OnD3D10CreateDevice(device));
    V_RETURN(settingsDialog.OnD3D10CreateDevice(device));

    V_RETURN(D3DX10CreateFont(device, 15, 0, FW_BOLD, 1, FALSE, DEFAULT_CHARSET,
                              OUT_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE,
                              L"Arial", &font));
    V_RETURN(D3DX10CreateSprite(device, 512, &sprite));
    txtHelper = new CDXUTTextHelper(NULL, NULL, font, sprite, 15);

    timer = new Timer(device);
    timer->setEnabled(hud.GetCheckBox(IDC_PROFILE)->GetChecked());

    Copy::init(device);

    loadImage();
    if (testDepthView != NULL) {
        hud.GetComboBox(IDC_DETECTIONMODE)->SetSelectedByIndex(1);
    }

    return S_OK;
}


void CALLBACK onDestroyDevice(void *context) {
    dialogResourceManager.OnD3D10DestroyDevice();
    settingsDialog.OnD3D10DestroyDevice();
    DXUTGetGlobalResourceCache().OnDestroyDevice();

    SAFE_RELEASE(font);
    SAFE_RELEASE(sprite);
    SAFE_DELETE(txtHelper);

    SAFE_DELETE(timer);

    Copy::release();
    
    SAFE_RELEASE(testView);
    SAFE_RELEASE(testDepthView);
}


HRESULT CALLBACK onResizedSwapChain(ID3D10Device *device, IDXGISwapChain *swapChain, const DXGI_SURFACE_DESC *desc, void *context) {
    HRESULT hr;
    V_RETURN(dialogResourceManager.OnD3D10ResizedSwapChain(device, desc));
    V_RETURN(settingsDialog.OnD3D10ResizedSwapChain(device, desc));

    D3DX10_IMAGE_INFO info;
    WCHAR *text = hud.GetComboBox(IDC_INPUT)->GetSelectedItem()->strText;
    if (int(hud.GetComboBox(IDC_INPUT)->GetSelectedData()) != -1) {
        V(D3DX10GetImageInfoFromResource(GetModuleHandle(NULL), text, NULL, &info, NULL));
    } else {
        V(D3DX10GetImageInfoFromFile(text, NULL, &info, NULL));
    }

    mainRenderTarget = new RenderTarget(device, info.Width, info.Height, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB);
    mlaa = new MLAA(device, info.Width, info.Height);
    depthStencil = new DepthStencil(device, info.Width, info.Height,  DXGI_FORMAT_R24G8_TYPELESS, DXGI_FORMAT_D24_UNORM_S8_UINT, DXGI_FORMAT_R24_UNORM_X8_TYPELESS);
    DXUTGetDXGISwapChain()->GetBuffer(0, __uuidof(backbufferTexture), reinterpret_cast<void**>(&backbufferTexture));

    hud.SetLocation(desc->Width - (45 + HUD_WIDTH), 0);

    return S_OK;
}


void CALLBACK onReleasingSwapChain(void *context) {
    dialogResourceManager.OnD3D10ReleasingSwapChain();

    SAFE_DELETE(mainRenderTarget);
    SAFE_DELETE(mlaa);
    SAFE_DELETE(depthStencil);
    SAFE_RELEASE(backbufferTexture);
}


void drawHud(ID3D10Device *device, float elapsedTime) {
    if (showHud) {
        ID3D10RenderTargetView *backbufferView = DXUTGetD3D10RenderTargetView();
        device->OMSetRenderTargets(1, &backbufferView, NULL);
        hud.OnRender(elapsedTime);

        txtHelper->Begin();

        txtHelper->SetInsertionPos(2, 0);
        txtHelper->SetForegroundColor(D3DXCOLOR(1.0f, 1.0f, 1.0f, 1.0f));
        txtHelper->DrawTextLine(DXUTGetFrameStats(DXUTIsVsyncEnabled()));
        txtHelper->DrawTextLine(DXUTGetDeviceStats());
        txtHelper->DrawTextLine(L"Press 'tab' to toogle the HUD, 'a' and 'd' to quickly cycle through the images");

        if (timer->isEnabled()) {
            wstringstream s;
            s << setprecision(5) << std::fixed;
            s << *timer;
            txtHelper->DrawTextLine(s.str().c_str());
        }

        txtHelper->End();
    }
}


void drawTextures(ID3D10Device *device) {
    ID3D10RenderTargetView *backbufferView = DXUTGetD3D10RenderTargetView();
    switch (int(hud.GetComboBox(IDC_VIEWMODE)->GetSelectedData())) {
        case 1:
            Copy::go(*mlaa->getEdgeRenderTarget(), backbufferView);
            break;
        case 2:
            Copy::go(*mlaa->getBlendRenderTarget(), backbufferView);
            break;
        default:
            break;
    }
}


void CALLBACK onFrameRender(ID3D10Device *device, double time, float elapsedTime, void *context) {
    if (settingsDialog.IsActive()) {
        settingsDialog.OnRender(elapsedTime);
        return;
    }

    ID3D10RenderTargetView *backbufferView = DXUTGetD3D10RenderTargetView();
    
    float clearColor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    device->ClearRenderTargetView(backbufferView, clearColor);
    device->ClearDepthStencilView(*depthStencil, D3D10_CLEAR_STENCIL, 1.0, 0);

    if (hud.GetCheckBox(IDC_ANTIALIASING)->GetChecked()) {
        /**
         * We are not taking into account the buffer copying times because in a
         * conventional situation (working with a real scene instead of an
         * image), they could be easily avoided.
         */

        Copy::go(testView, *mainRenderTarget);
        bool useDepth = int(hud.GetComboBox(IDC_DETECTIONMODE)->GetSelectedData()) == 1;

        timer->start();
        if (useDepth) {
            mlaa->go(testView, *mainRenderTarget, *depthStencil, testDepthView);
        } else {
            mlaa->go(testView, *mainRenderTarget, *depthStencil);
        }
        timer->clock(L"MLAA");

        ID3D10RenderTargetView *view = *mainRenderTarget;
        D3D10_VIEWPORT viewport = Utils::viewportFromView(view);
        Copy::go(*mainRenderTarget, backbufferView, &viewport);
    } else {
        ID3D10RenderTargetView *view = *mainRenderTarget;
        D3D10_VIEWPORT viewport = Utils::viewportFromView(view);
        Copy::go(testView, backbufferView, &viewport);
    }
    
    drawTextures(device);
    drawHud(device, elapsedTime);
}


LRESULT CALLBACK msgProc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam, bool *finished, void *context) {
    *finished = dialogResourceManager.MsgProc(hwnd, msg, wparam, lparam);
    if (*finished) {
        return 0;
    }

    if (settingsDialog.IsActive()) {
        settingsDialog.MsgProc(hwnd, msg, wparam, lparam);
        return 0;
    }

    *finished = hud.MsgProc(hwnd, msg, wparam, lparam);
    if (*finished) {
        return 0;
    }

    return 0;
}


void CALLBACK keyboardProc(UINT nchar, bool keyDown, bool altDown, void *context) {
    switch (nchar) {
        case VK_TAB: {
            if (keyDown) {
                showHud = !showHud;
            }
            break;
        }
        case 'A': {
            if (keyDown) {
                int previous = hud.GetComboBox(IDC_INPUT)->GetSelectedIndex() - 1;
                hud.GetComboBox(IDC_INPUT)->SetSelectedByIndex(previous > 0? previous : 0);
                loadImage();
            }
            break;
        }          
        case 'D': {
            if (keyDown) {
                int next = hud.GetComboBox(IDC_INPUT)->GetSelectedIndex() + 1;
                int n = hud.GetComboBox(IDC_INPUT)->GetNumItems();
                hud.GetComboBox(IDC_INPUT)->SetSelectedByIndex(next < n? next : n);
                loadImage();
            }
            break;
        }
    }
}


bool CALLBACK modifyDeviceSettings(DXUTDeviceSettings *settings, void *context) {
    settingsDialog.GetDialogControl()->GetComboBox(DXUTSETTINGSDLG_D3D10_MULTISAMPLE_COUNT)->SetEnabled(false);
    settingsDialog.GetDialogControl()->GetComboBox(DXUTSETTINGSDLG_D3D10_MULTISAMPLE_QUALITY)->SetEnabled(false);
    settingsDialog.GetDialogControl()->GetStatic(DXUTSETTINGSDLG_D3D10_MULTISAMPLE_COUNT_LABEL)->SetEnabled(false);
    settingsDialog.GetDialogControl()->GetStatic(DXUTSETTINGSDLG_D3D10_MULTISAMPLE_QUALITY_LABEL)->SetEnabled(false);
    settings->d3d10.AutoCreateDepthStencil = false;
    return true;
}


void buildInputComboBox() {
    hud.GetComboBox(IDC_INPUT)->RemoveAllItems();

    for (int i = 0; i < 7; i++) {
        wstringstream s;
        s.fill('0');
        s << L"Unigine" << setw(2) << i + 1 << ".png";
        hud.GetComboBox(IDC_INPUT)->AddItem(s.str().c_str(), NULL);
    }

    WIN32_FIND_DATA data;
    HANDLE handle = FindFirstFile(L"Images\\*.png", &data);
    if (handle != INVALID_HANDLE_VALUE){
        do {
            hud.GetComboBox(IDC_INPUT)->AddItem((L"Images\\" + wstring(data.cFileName)).c_str(), (LPVOID) -1);
        } while(FindNextFile(handle, &data));
        FindClose(handle);
    }
}


void CALLBACK onGUIEvent(UINT event, int id, CDXUTControl *control, void *context) {
    switch (id) {
        case IDC_TOGGLEFULLSCREEN:
            DXUTToggleFullScreen();
            break;
        case IDC_CHANGEDEVICE:
            settingsDialog.SetActive(!settingsDialog.IsActive());
            break;
        case IDC_LOADIMAGE: {
            OPENFILENAME params;
            ZeroMemory(&params, sizeof(OPENFILENAME));
            
            const int maxLength = 512;
            WCHAR file[maxLength]; 
            file[0] = 0;

            params.lStructSize   = sizeof(OPENFILENAME);
            params.hwndOwner     = DXUTGetHWND();
            params.lpstrFilter   = L"Image Files (*.bmp, *.jpg, *.png)\0*.bmp;*.jpg;*.png\0\0";
            params.nFilterIndex  = 1;
            params.lpstrFile     = file;
            params.nMaxFile      = maxLength;
            params.lpstrTitle    = L"Open File";
            params.lpstrInitialDir = file;
            params.nMaxFileTitle = sizeof(params.lpstrTitle);
            params.Flags         = OFN_FILEMUSTEXIST;
            params.dwReserved    = OFN_EXPLORER;

            if (GetOpenFileName(&params)) {
                buildInputComboBox();
                hud.GetComboBox(IDC_INPUT)->AddItem(file, (LPVOID) -1);
                hud.GetComboBox(IDC_INPUT)->SetSelectedByData((LPVOID) -1);

                loadImage();
                resizeWindow();
                onReleasingSwapChain(NULL);
                onResizedSwapChain(DXUTGetD3D10Device(), DXUTGetDXGISwapChain(), DXUTGetDXGIBackBufferSurfaceDesc(), NULL);
            }
        }
        case IDC_VIEWMODE:
            if (event == EVENT_COMBOBOX_SELECTION_CHANGED) {
                if (int(hud.GetComboBox(IDC_VIEWMODE)->GetSelectedData()) > 0) {
                    hud.GetCheckBox(IDC_ANTIALIASING)->SetChecked(true);
                }
            }
            break;
        case IDC_INPUT:
            if (event == EVENT_COMBOBOX_SELECTION_CHANGED) {
                timer->reset();            
                loadImage();
                resizeWindow();
                onReleasingSwapChain(NULL);
                onResizedSwapChain(DXUTGetD3D10Device(), DXUTGetDXGISwapChain(), DXUTGetDXGIBackBufferSurfaceDesc(), NULL);
            }
            break;
        case IDC_ANTIALIASING:
            if (event == EVENT_CHECKBOX_CHANGED) {
                timer->reset();
                hud.GetComboBox(IDC_VIEWMODE)->SetSelectedByIndex(0);
            }
            break;
        case IDC_PROFILE:
            if (event == EVENT_CHECKBOX_CHANGED) {
                timer->reset();
                timer->setEnabled(hud.GetCheckBox(IDC_PROFILE)->GetChecked());
            }
            break;
        case IDC_MAXSEARCHSTEPS:
            if (event == EVENT_SLIDER_VALUE_CHANGED) {
                CDXUTSlider *slider = (CDXUTSlider *) control;
                int min, max;
                slider->GetRange(min, max);

                float scale = float(slider->GetValue()) / (max - min);
                mlaa->setMaxSearchSteps(scale * 32.0f);
            
                wstringstream s;
                s << L"Max Search Steps: " << scale * 32.0f;
                hud.GetStatic(IDC_MAXSEARCHSTEPS_LABEL)->SetText(s.str().c_str());
            }
            break;
        case IDC_THRESHOLD:
            if (event == EVENT_SLIDER_VALUE_CHANGED) {
                CDXUTSlider *slider = (CDXUTSlider *) control;
                int min, max;
                slider->GetRange(min, max);

                float scale = float(slider->GetValue()) / (max - min);
                mlaa->setThreshold(scale * 0.5f);
            
                wstringstream s;
                s << L"Threshold: " << scale * 0.5f;
                hud.GetStatic(IDC_THRESHOLD_LABEL)->SetText(s.str().c_str());
            }
            break;
        default:
            break;
    }
}


void initApp() {
    settingsDialog.Init(&dialogResourceManager);
    hud.Init(&dialogResourceManager);
    hud.SetCallback(onGUIEvent);

    int iY = 10;

    hud.AddButton(IDC_TOGGLEFULLSCREEN, L"Toggle full screen", 35, iY, HUD_WIDTH, 22);
    hud.AddButton(IDC_CHANGEDEVICE, L"Change device", 35, iY += 24, HUD_WIDTH, 22, VK_F2);
    hud.AddButton(IDC_LOADIMAGE, L"Load image", 35, iY += 24, HUD_WIDTH, 22);
    
    hud.AddComboBox(IDC_DETECTIONMODE, 35, iY += 24, HUD_WIDTH, 22, 0, false);
    hud.GetComboBox(IDC_DETECTIONMODE)->AddItem(L"Color edge det.", (LPVOID) 0);
    hud.GetComboBox(IDC_DETECTIONMODE)->AddItem(L"Depth edge det.", (LPVOID) 1);

    hud.AddComboBox(IDC_VIEWMODE, 35, iY += 24, HUD_WIDTH, 22, 0, false);
    hud.GetComboBox(IDC_VIEWMODE)->AddItem(L"View image", (LPVOID) 0);
    hud.GetComboBox(IDC_VIEWMODE)->AddItem(L"View edges", (LPVOID) 1);
    hud.GetComboBox(IDC_VIEWMODE)->AddItem(L"View weights", (LPVOID) 2);

    hud.AddComboBox(IDC_INPUT, 35, iY += 24, HUD_WIDTH, 22, 0, false);
    buildInputComboBox();
    
    hud.AddCheckBox(IDC_ANTIALIASING, L"MLAA Anti-Aliasing", 35, iY += 24, HUD_WIDTH, 22, true);
    hud.AddCheckBox(IDC_PROFILE, L"Profile", 35, iY += 24, HUD_WIDTH, 22, false);

    hud.AddStatic(IDC_MAXSEARCHSTEPS_LABEL, L"Max Search Steps: 4.0", 35, iY += 24, HUD_WIDTH, 22);
    hud.AddSlider(IDC_MAXSEARCHSTEPS, 35, iY += 24, HUD_WIDTH, 22, 0, 100, int(100.0f * 4.0f / 32.0f));

    hud.AddStatic(IDC_THRESHOLD_LABEL, L"Threshold: 0.1", 35, iY += 24, HUD_WIDTH, 22);
    hud.AddSlider(IDC_THRESHOLD, 35, iY += 24, HUD_WIDTH, 22, 0, 100, int(100.0f * 0.1f / 0.5f));
}


INT WINAPI wWinMain(HINSTANCE, HINSTANCE, LPWSTR, int) {
    // Enable run-time memory check for debug builds.
    #if defined(DEBUG) | defined(_DEBUG)
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    #endif

    DXUTSetCallbackD3D10DeviceAcceptable(isDeviceAcceptable);
    DXUTSetCallbackD3D10DeviceCreated(onCreateDevice);
    DXUTSetCallbackD3D10DeviceDestroyed(onDestroyDevice);
    DXUTSetCallbackD3D10SwapChainResized(onResizedSwapChain);
    DXUTSetCallbackD3D10SwapChainReleasing(onReleasingSwapChain);
    DXUTSetCallbackD3D10FrameRender(onFrameRender);

    DXUTSetCallbackMsgProc(msgProc);
    DXUTSetCallbackKeyboard(keyboardProc);
    DXUTSetCallbackDeviceChanging(modifyDeviceSettings);

    initApp();

    if (FAILED(DXUTInit(true, true, L"-forcevsync:0"))) {
		return -1;
	}

    DXUTSetCursorSettings(true, true);
    if (FAILED(DXUTCreateWindow(L"Practical Morphological Anti-Aliasing Demo"))) {
		return -1;
	}

    if (FAILED(DXUTCreateDevice(true, 1280, 720))) {
		return -1;
	}
    
    resizeWindow();

    DXUTMainLoop();

    return DXUTGetExitCode();
}
