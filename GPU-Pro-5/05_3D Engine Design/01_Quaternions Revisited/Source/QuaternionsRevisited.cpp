/****************************************************************************

  GPU Pro 5 : Quaternions revisited - sample code
  All sample code written from scratch by Sergey Makeev specially for article.

  Copyright (c) 2013, Sergey Makeev

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software.

  2. If you use this software in a non-commercial product, an acknowledgment
     in the product documentation would be appreciated but is not required.

  3. If you use this software in a commercial product you requried to credit
     the author.

  4. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.

  5. This notice may not be removed or altered from any source distribution.
    

   Please let me know if you use this code in your products or have any questions or suggestions.

   e-mail: sergeymakeev@inbox.ru
   http://linkedin.com/in/sergeymakeev/

****************************************************************************/
#include "DXUT.h"
#include "DXUTgui.h"
#include "DXUTmisc.h"
#include "DXUTSettingsDlg.h"
#include "SDKmisc.h"
#include "resource.h"

#include <fbxsdk.h>
#include <vector>
#include <set>
#include "Assert.h"
#include "FBXImport.h"
#include "D3DMesh.h"
#include "D3DAnimation.h"
#include "D3DFloor.h"
#include "Camera.h"


//--------------------------------------------------------------------------------------
// UI control IDs
//--------------------------------------------------------------------------------------
#define IDC_TOGGLEFULLSCREEN    1
#define IDC_TOGGLEREF           2
#define IDC_CHANGEDEVICE        3
#define IDC_DISABLEALBEDO       4
#define IDC_PAUSEANIMATION      5
#define IDC_SHOWNORMALS         6
#define IDC_TECHNIQUECOMBO      7
#define IDC_DISABLESKINING      8

// Manager for shared resources of dialogs
CDXUTDialogResourceManager g_DialogResourceManager; 

// Device settings dialog
CD3DSettingsDlg g_SettingsDlg;

// Dialog for standard controls
CDXUTDialog g_HUD;

// Dialog for sample specific controls
CDXUTDialog g_SampleUI;

CDXUTTextHelper* g_pTxtHelper = NULL;

ID3DXFont* g_pFont9 = NULL;

ID3DXSprite* g_pSprite9 = NULL;


bool g_DisableAlbedo = false;
bool g_PauseAnimation = false;
bool g_ShowNormals = false;
bool g_DisableSkining = false;
int g_TechniqueIndex = 3;

// Mesh and animation
D3DMesh d3dMesh;
D3DAnimation d3dAnimation;

// Simple floor
D3DFloor d3dFloor;

// Free fly camera
FreeCamera freeCamera;



//--------------------------------------------------------------------------------------
// Rejects any D3D9 devices that aren't acceptable to the app by returning false
//--------------------------------------------------------------------------------------
bool CALLBACK IsD3D9DeviceAcceptable( D3DCAPS9* pCaps, D3DFORMAT AdapterFormat, D3DFORMAT BackBufferFormat, bool bWindowed, void* pUserContext )
{
	// Typically want to skip back buffer formats that don't support alpha blending
	IDirect3D9* pD3D = DXUTGetD3D9Object();
	if( FAILED( pD3D->CheckDeviceFormat( pCaps->AdapterOrdinal, pCaps->DeviceType, AdapterFormat, D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING, D3DRTYPE_TEXTURE, BackBufferFormat ) ) )
	{
		return false;
	}
	return true;
}

//--------------------------------------------------------------------------------------
// Handles the GUI events
//--------------------------------------------------------------------------------------
void CALLBACK OnGUIEvent( UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext )
{
	switch( nControlID )
	{
	case IDC_TOGGLEFULLSCREEN:
		DXUTToggleFullScreen();
		break;
	case IDC_TOGGLEREF:
		DXUTToggleREF();
		break;
	case IDC_CHANGEDEVICE:
		g_SettingsDlg.SetActive( !g_SettingsDlg.IsActive() );
		break;
	case IDC_DISABLEALBEDO:
		g_DisableAlbedo = ((CDXUTCheckBox*)pControl )->GetChecked();
		break;
	case IDC_PAUSEANIMATION:
		g_PauseAnimation = ((CDXUTCheckBox*)pControl )->GetChecked();
		break;
	case IDC_SHOWNORMALS:
		g_ShowNormals = ((CDXUTCheckBox*)pControl )->GetChecked();
		break;
	case IDC_DISABLESKINING:
		g_DisableSkining = ((CDXUTCheckBox*)pControl )->GetChecked();
		break;
	case IDC_TECHNIQUECOMBO:
		g_TechniqueIndex = ((CDXUTComboBox* )pControl)->GetSelectedIndex();
		break;
		
		
	}
}


//--------------------------------------------------------------------------------------
// Before a device is created, modify the device settings as needed
//--------------------------------------------------------------------------------------
bool CALLBACK ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, void* pUserContext )
{
	// Turn vsync off
	pDeviceSettings->d3d9.pp.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;
	return true;
}


//--------------------------------------------------------------------------------------
// Create any D3D9 resources that will live through a device reset (D3DPOOL_MANAGED)
// and aren't tied to the back buffer size
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D9CreateDevice( IDirect3DDevice9* pd3dDevice, const D3DSURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext )
{
	g_SettingsDlg.Init( &g_DialogResourceManager );
	g_HUD.Init( &g_DialogResourceManager );
	g_SampleUI.Init( &g_DialogResourceManager );

	g_DialogResourceManager.OnD3D9CreateDevice( pd3dDevice );
	g_SettingsDlg.OnD3D9CreateDevice( pd3dDevice );


	g_HUD.SetCallback( OnGUIEvent ); 
	int iY = 10;
	g_HUD.AddButton( IDC_TOGGLEFULLSCREEN, L"Toggle full screen", 35, iY, 125, 22 );
	g_HUD.AddButton( IDC_TOGGLEREF, L"Toggle REF (F3)", 35, iY += 24, 125, 22, VK_F3 );
	g_HUD.AddButton( IDC_CHANGEDEVICE, L"Change device (F2)", 35, iY += 24, 125, 22, VK_F2 );

	g_SampleUI.SetCallback( OnGUIEvent );

	iY = 10;
	g_SampleUI.AddCheckBox( IDC_DISABLEALBEDO, L"Disable albedo", 35, iY += 24, 160, 22, g_DisableAlbedo );
	g_SampleUI.AddCheckBox( IDC_PAUSEANIMATION, L"Pause animation", 35, iY += 24, 160, 22, g_PauseAnimation );
	g_SampleUI.AddCheckBox( IDC_DISABLESKINING, L"Disable skining", 35, iY += 24, 160, 22, g_DisableSkining );
	g_SampleUI.AddCheckBox( IDC_SHOWNORMALS, L"Show normals", 35, iY += 24, 160, 22, g_ShowNormals );


	CDXUTComboBox* pCombo;
	g_SampleUI.AddComboBox( IDC_TECHNIQUECOMBO, 35, iY += 30, 160, 22, 0, false, &pCombo );
	if( pCombo )
	{
		pCombo->SetDropHeight( 100 );
		pCombo->AddItem( L"Unpacked TBN", ( LPVOID )0x0 );
		pCombo->AddItem( L"Packed TBN", ( LPVOID )0x1 );
		pCombo->AddItem( L"Unpacked Quaternion", ( LPVOID )0x2 );
		pCombo->AddItem( L"Packed Quaternion", ( LPVOID )0x3 );
		pCombo->SetSelectedByIndex(g_TechniqueIndex);
	}

	iY = 10;

	D3DXCreateFont( pd3dDevice, 15, 0, FW_BOLD, 1, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Arial", &g_pFont9 );

	d3dAnimation.Create(pd3dDevice);
	d3dMesh.Create(pd3dDevice);
	d3dFloor.Create(pd3dDevice);

	FBXImporter importer;
	const char* fbxFileName = ".\\data\\MilitaryMechanic.fbx";
	importer.Import(fbxFileName, &d3dMesh, &d3dAnimation);

	return S_OK;
}


//--------------------------------------------------------------------------------------
// Create any D3D9 resources that won't live through a device reset (D3DPOOL_DEFAULT) 
// or that are tied to the back buffer size 
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D9ResetDevice( IDirect3DDevice9* pd3dDevice, const D3DSURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext )
{
	if( g_pFont9 )
		g_pFont9->OnResetDevice();

	D3DXCreateSprite( pd3dDevice, &g_pSprite9 );

	g_pTxtHelper = new CDXUTTextHelper( g_pFont9, g_pSprite9, NULL, NULL, 15 );

	g_DialogResourceManager.OnD3D9ResetDevice();
	g_SettingsDlg.OnD3D9ResetDevice();

	d3dMesh.DeviceReset();
	d3dAnimation.DeviceReset();

	g_HUD.SetLocation( pBackBufferSurfaceDesc->Width - 170, 0 );
	g_HUD.SetSize( 170, 60 );
	g_SampleUI.SetLocation( pBackBufferSurfaceDesc->Width - 200, 65 );
	g_SampleUI.SetSize( 170, 300 );

	return S_OK;
}


//--------------------------------------------------------------------------------------
// Handle updates to the scene.  This is called regardless of which D3D API is used
//--------------------------------------------------------------------------------------
void CALLBACK OnFrameMove( double fTime, float fElapsedTime, void* pUserContext )
{
}


//--------------------------------------------------------------------------------------
// Render the scene using the D3D9 device
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D9FrameRender( IDirect3DDevice9* pd3dDevice, double fTime, float fElapsedTime, void* pUserContext )
{
	// If the settings dialog is being shown, then render it instead of rendering the app's scene
	if( g_SettingsDlg.IsActive() )
	{
		g_SettingsDlg.OnRender( fElapsedTime );
		return;
	}

	HRESULT hr;

	// Clear the render target and the zbuffer 
	V( pd3dDevice->Clear( 0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 0x8dadd4, 1.0f, 0 ) );

	// Render the scene
	if( SUCCEEDED( pd3dDevice->BeginScene() ) )
	{
		freeCamera.Update( fElapsedTime );
		D3DXMATRIXA16 mtxView = freeCamera.GetView();

		// Get back buffer description and determine aspect ratio
		const D3DSURFACE_DESC* pBackBufferSurfaceDesc = DXUTGetD3D9BackBufferSurfaceDesc();
		float Width = (float)pBackBufferSurfaceDesc->Width;
		float Height = (float)pBackBufferSurfaceDesc->Height;
		float fAspectRatio = Width / Height;

		float fovHorizontal = D3DX_PI / 4.0f;
		float nearZ = 0.05f;
		float farZ = 500.0f;

		D3DXMATRIXA16 mtxProj;
		D3DXMatrixPerspectiveFovLH( &mtxProj, fovHorizontal, fAspectRatio, nearZ, farZ );

		// Convert projection matrix to right handed coordinate system
		D3DXMATRIX mtxMirror;
		D3DXMatrixScaling(&mtxMirror, -1.0, 1.0, 1.0);
		D3DXMatrixMultiply( &mtxProj, &mtxMirror, &mtxProj);

		// Build view projection matrix
		D3DXMATRIX mtxViewProj;
		D3DXMatrixMultiply( &mtxViewProj, &mtxView, &mtxProj );

		d3dFloor.Draw(mtxViewProj);
		d3dMesh.Draw(mtxViewProj, freeCamera.GetForward(), fElapsedTime);

		g_pTxtHelper->Begin();
		g_pTxtHelper->SetInsertionPos( 5, 5 );
		g_pTxtHelper->SetForegroundColor( D3DXCOLOR( 1.0f, 1.0f, 0.0f, 1.0f ) );
		g_pTxtHelper->DrawTextLine( DXUTGetFrameStats( DXUTIsVsyncEnabled() ) );
		g_pTxtHelper->DrawTextLine( DXUTGetDeviceStats() );
		
		g_pTxtHelper->SetForegroundColor( D3DXCOLOR( 1.0f, 0.9f, 0.0f, 1.0f ) );

		float frameTimeMs = DXUTGetElapsedTime() * 1000.0f;




		g_pTxtHelper->DrawFormattedTextLine(L"\n\n"
			L"Frame time : %3.2f ms\n\n"
			L"Navigation:\n"
			L" Click and Hold Right Mouse Button (RMB) and move the mouse to look around\n"
			L" As you hold Right Mouse Button (RMB) use the W A S D to movement", frameTimeMs);

		g_pTxtHelper->SetForegroundColor( D3DXCOLOR( 1.0f, 1.0f, 1.0f, 1.0f ) );
		g_pTxtHelper->SetInsertionPos((int)Width - 350, 10);

		if (g_DisableSkining)
		{
			if (g_TechniqueIndex == 0)
			{
				g_pTxtHelper->DrawTextLine (L"Vertex size 56 bytes\n\n"
					L"struct Vertex\n"
					L"{\n"
					L"   float3 pos;\n"
					L"   float2 uv;\n"
					L"   float3 tangent;\n"
					L"   float3 normal;\n"
					L"   float3 binormal;\n"
					L"};\n");
			} else
			{
				if (g_TechniqueIndex == 1)
				{
					g_pTxtHelper->DrawTextLine (L"Vertex size 28 bytes\n\n"
						L"struct Vertex\n"
						L"{\n"
						L"   float3 pos;\n"
						L"   short2 uv;\n"
						L"   ubyte4 tangent;\n"
						L"   ubyte4 normal;\n"
						L"   ubyte4 binormal;\n"
						L"};\n");
				} else
				{
					if (g_TechniqueIndex == 2)
					{
						g_pTxtHelper->DrawTextLine (L"Vertex size 32 bytes\n\n"
							L"struct Vertex\n"
							L"{\n"
							L"   float3 pos;\n"
							L"   short2 uv;\n"
							L"   float4 tbn;\n"
							L"};\n");
					} else
					{
						if (g_TechniqueIndex == 3)
						{
							g_pTxtHelper->DrawTextLine (L"Vertex size 20 bytes\n\n"
								L"struct Vertex\n"
								L"{\n"
								L"   float3 pos;\n"
								L"   short2 uv;\n"
								L"   ubyte4 tbn;\n"
								L"};\n");
						}
					}
				}
			}
		} else
		{
			if (g_TechniqueIndex == 0)
			{
				g_pTxtHelper->DrawTextLine (L"Vertex size 76 bytes\n\n"
					L"struct Vertex\n"
					L"{\n"
					L"   float3 pos;\n"
					L"   float2 uv;\n"
					L"   float3 tangent;\n"
					L"   float3 normal;\n"
					L"   float3 binormal;\n"
					L"   float4 weights;\n"
					L"   ubyte4 indices;\n"
					L"};\n");
			} else
			{
				if (g_TechniqueIndex == 1)
				{
					g_pTxtHelper->DrawTextLine (L"Vertex size 40 bytes\n\n"
						L"struct Vertex\n"
						L"{\n"
						L"   float3 pos;\n"
						L"   short2 uv;\n"
						L"   ubyte4 tangent;\n"
						L"   ubyte4 normal;\n"
						L"   ubyte4 binormal;\n"
						L"   short4 weights;\n"
						L"   ubyte4 indices;\n"
						L"};\n");
				} else
				{
					if (g_TechniqueIndex == 2)
					{
						g_pTxtHelper->DrawTextLine (L"Vertex size 44 bytes\n\n"
							L"struct Vertex\n"
							L"{\n"
							L"   float3 pos;\n"
							L"   short2 uv;\n"
							L"   float4 tbn;\n"
							L"   short4 weights;\n"
							L"   ubyte4 indices;\n"
							L"};\n");
					} else
					{
						if (g_TechniqueIndex == 3)
						{
							g_pTxtHelper->DrawTextLine (L"Vertex size 32 bytes\n\n"
								L"struct Vertex\n"
								L"{\n"
								L"   float3 pos;\n"
								L"   short2 uv;\n"
								L"   ubyte4 tbn;\n"
								L"   short4 weights;\n"
								L"   ubyte4 indices;\n"
								L"};\n");
						}
					}
				}
			}
		}


		g_pTxtHelper->End();

		V( g_HUD.OnRender( fElapsedTime ) );
		V( g_SampleUI.OnRender( fElapsedTime ) );
		V( pd3dDevice->EndScene() );
	}
}

//--------------------------------------------------------------------------------------
// Handle messages to the application 
//--------------------------------------------------------------------------------------
LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing, void* pUserContext )
{
	// Pass messages to dialog resource manager calls so GUI state is updated correctly
	*pbNoFurtherProcessing = g_DialogResourceManager.MsgProc( hWnd, uMsg, wParam, lParam );
	if( *pbNoFurtherProcessing )
		return 0;

	// Pass messages to settings dialog if its active
	if( g_SettingsDlg.IsActive() )
	{
		g_SettingsDlg.MsgProc( hWnd, uMsg, wParam, lParam );
		return 0;
	}

	// Give the dialogs a chance to handle the message first
	*pbNoFurtherProcessing = g_HUD.MsgProc( hWnd, uMsg, wParam, lParam );
	if( *pbNoFurtherProcessing )
		return 0;

	*pbNoFurtherProcessing = g_SampleUI.MsgProc( hWnd, uMsg, wParam, lParam );
	if( *pbNoFurtherProcessing )
		return 0;

	return 0;
}


//--------------------------------------------------------------------------------------
// Release D3D9 resources created in the OnD3D9ResetDevice callback 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D9LostDevice( void* pUserContext )
{

	if( g_pFont9 )
		g_pFont9->OnLostDevice();

	SAFE_RELEASE( g_pSprite9 );
	SAFE_DELETE( g_pTxtHelper );

	g_DialogResourceManager.OnD3D9LostDevice();
	g_SettingsDlg.OnD3D9LostDevice();

	d3dMesh.DeviceLost();
	d3dAnimation.DeviceLost();
}


//--------------------------------------------------------------------------------------
// Release D3D9 resources created in the OnD3D9CreateDevice callback 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D9DestroyDevice( void* pUserContext )
{
	g_DialogResourceManager.OnD3D9DestroyDevice();
	g_SettingsDlg.OnD3D9DestroyDevice();

	d3dAnimation.Destroy();
	d3dMesh.Destroy();
	d3dFloor.Destroy();

	SAFE_DELETE(g_pTxtHelper);
	SAFE_RELEASE(g_pFont9);
	SAFE_RELEASE(g_pSprite9);
}



//--------------------------------------------------------------------------------------
// Initialize everything and go into a render loop
//--------------------------------------------------------------------------------------
INT WINAPI wWinMain( HINSTANCE, HINSTANCE, LPWSTR lpCmdLine, int nShowCmd)
{
	wchar_t executableFullPath[MAX_PATH];
	GetModuleFileName( NULL, executableFullPath, MAX_PATH );

	wchar_t executableDrive[MAX_PATH];
	wchar_t executableDir[MAX_PATH];
	_wsplitpath_s(executableFullPath, executableDrive, MAX_PATH, executableDir, MAX_PATH, 0, 0, 0, 0);

	wchar_t currentDir[MAX_PATH] = { L'\0' };
	wcscat_s(currentDir, executableDrive);
	wcscat_s(currentDir, executableDir);
	wcscat_s(currentDir, L"..\\");

	BOOL result = SetCurrentDirectory(currentDir);
	ASSERT(result != FALSE, "Can't change current directory.");

	// Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
#endif

	// Set the callback functions
	DXUTSetCallbackD3D9DeviceAcceptable( IsD3D9DeviceAcceptable );
	DXUTSetCallbackD3D9DeviceCreated( OnD3D9CreateDevice );
	DXUTSetCallbackD3D9DeviceReset( OnD3D9ResetDevice );
	DXUTSetCallbackD3D9FrameRender( OnD3D9FrameRender );
	DXUTSetCallbackD3D9DeviceLost( OnD3D9LostDevice );
	DXUTSetCallbackD3D9DeviceDestroyed( OnD3D9DestroyDevice );
	DXUTSetCallbackDeviceChanging( ModifyDeviceSettings );
	DXUTSetCallbackMsgProc( MsgProc );
	DXUTSetCallbackFrameMove( OnFrameMove );

	// Initialize DXUT and create the desired Win32 window and Direct3D device for the application
	
	// Parse the command line and show msgboxes
	DXUTInit( true, true );
	
	// Handle the default hotkeys
	DXUTSetHotkeyHandling( true, true, true );
	
	// Show the cursor and clip it when in full screen
	DXUTSetCursorSettings( true, true ); 

	DXUTCreateWindow(L"GPU Pro 5 : 'Quaternions revisited'");
	DXUTCreateDevice(true, 1280, 720);

	// Start the render loop
	DXUTMainLoop();

	return DXUTGetExitCode();
}


