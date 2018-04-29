//--------------------------------------------------------------------------------------
// File: OIT11LinkedLists.cpp
//
// This sample implements OIT via linked lists using DirectX 11.
// It accompanies the chapter "Order-Independent Transparency Using Per-Pixel 
// Linked Lists" from the book GPU Pro 2.
// 
//--------------------------------------------------------------------------------------
#include <afxpriv.h>
#include "strsafe.h"
#include "DXUT.h"
#include "DXUTcamera.h"
#include "DXUTgui.h"
#include "DXUTsettingsDlg.h"
#include "SDKmisc.h"
#include "SDKMesh.h"
#include <D3DX11tex.h>
#include <D3DX11.h>
#include <D3DX11core.h>
#include <D3DX11async.h>
#include "resource.h"
#include "OIT11LinkedLists.h"


//--------------------------------------------------------------------------------------
// Macros
//--------------------------------------------------------------------------------------
#define FLOAT_POSITIVE_RANDOM(x)        ( ((x)*rand()) / RAND_MAX )
#define FLOAT_RANDOM(x)                 ((((2.0f*rand())/RAND_MAX) - 1.0f)*(x))
#define PI                              3.14159f
#define MAX_NUMBER_OF_CUBES             20

#define NUM_THREADS_X                   8
#define NUM_THREADS_Y                   8

// Defines the number of tiles to divide the screen dimensions by
// Tile mode is used for linked list method only, and is disabled by default
// (thus tile size is 1x1, i.e. fullscreen)
#define NUMBER_OF_TILES_X               3
#define NUMBER_OF_TILES_Y               2

//--------------------------------------------------------------------------------------
// Structures
//--------------------------------------------------------------------------------------
struct SIMPLEVERTEX 
{
        FLOAT x, y, z;
        FLOAT u, v;
};

struct EXTENDEDVERTEX 
{
        FLOAT x, y, z;
        FLOAT nx, ny, nz;
        FLOAT u, v;
};

struct GLOBAL_CONSTANT_BUFFER_STRUCT
{
    D3DXMATRIXA16    mView;
    D3DXMATRIXA16    mProjection;
    D3DXMATRIXA16    mViewProjection;
    D3DXVECTOR4      vLightVector;
    D3DXVECTOR4      vViewVector;
    D3DXVECTOR4      vScreenDimensions;
};

struct PER_MESH_CONSTANT_BUFFER_STRUCT
{
    D3DXMATRIXA16    mWorld;
    D3DXMATRIXA16    mWorldViewProjection;
    D3DXVECTOR4      vMeshColor;
};

struct TILE_COORDINATES_CONSTANT_BUFFER_STRUCT
{
    D3DXVECTOR4      vRectangleCoordinates;
    D3DXVECTOR4      vTileSize;
};

struct OBJECT_STRUCT
{
    D3DXVECTOR3      vPosition;
    D3DXVECTOR4      vOrientation;
    float            fScale;
    D3DXVECTOR4      vColor;
};

struct FRAGMENT_AND_LINK_BUFFER_STRUCT
{
    DWORD            vPixelColor;
    DWORD            uDepthAndCoverage;      // Coverage is only used when MSAA is enabled
    DWORD            dwNext;
};

struct MESH_DESCRIPTOR
{
    WCHAR*          sMeshName;
    WCHAR*          sMeshFileName;
    D3DXVECTOR3     vWorldSpacePosition;
    D3DXVECTOR4     vRotationAxisAndAngle;
    float           fScaleFactor;
    D3DXVECTOR4     vMeshColor;
    D3DXVECTOR4     vMeshColorRefraction;
};

struct BOUNDING_BOX_STRUCT
{
    D3DXVECTOR3     vPoint[8];
};

//--------------------------------------------------------------------------------------
// Enums
//--------------------------------------------------------------------------------------
enum
{
    OITMETHOD_NOSORTING,
    OITMETHOD_LINKEDLIST,
} eOITMethod;

//--------------------------------------------------------------------------------------
// Consts
//--------------------------------------------------------------------------------------
const SIMPLEVERTEX g_FullscreenQuad[4] = 
{
    { -1.0f, -1.0f, 0.0f, 0.0f, 1.0f },
    { -1.0f,  1.0f, 0.0f, 0.0f, 0.0f },
    {  1.0f, -1.0f, 0.0f, 1.0f, 1.0f },
    {  1.0f,  1.0f, 0.0f, 1.0f, 0.0f }
};

#define NUM_TRANSPARENT_MESHES 2
const MESH_DESCRIPTOR vTransparentMeshesInScene[NUM_TRANSPARENT_MESHES] =
{
    // sMeshName,   sMeshFileName,                    vWorldSpacePosition,                vRotationAxisAndAngle,                          fScaleFactor    vMeshColor                              vMeshColorRefraction
    L"Teapot",      L"media\\teapot\\teapot.sdkmesh", D3DXVECTOR3(0.0f, -2.1f,  0.0f),    D3DXVECTOR4(0.0f, 1.0f, 0.0f, 0.0f),            1.0f,           D3DXVECTOR4(0.2f, 0.2f, 1.0f, 0.5f),    D3DXVECTOR4(0.2f, 0.2f, 1.0f, 0.3f),
    L"Dwarf",       L"media\\dwarf\\dwarf.sdkmesh",   D3DXVECTOR3(0.0f, -2.99f, 0.0f),    D3DXVECTOR4(0.0f, 1.0f, 0.0f, 0.0f),            1.3f,           D3DXVECTOR4(1.0f, 1.0f, 1.0f, 0.35f),   D3DXVECTOR4(1.0f, 1.0f, 1.0f, 0.30f),
};

//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------

// DXUT resources
CDXUTDialogResourceManager  g_DialogResourceManager;    // manager for shared resources of dialogs
CFirstPersonCamera          g_Camera;                   // First-person camera for navigating the scene
CModelViewerCamera          g_LCamera;                  // A model viewing camera for the light
CD3DSettingsDlg             g_D3DSettingsDlg;           // Device settings dialog
CDXUTDialog                 g_HUD;                      // manages the 3D   
CDXUTDialog                 g_SampleUI;                 // dialog for sample specific controls
CDXUTTextHelper*            g_pTxtHelper = NULL;

// Object geometry
CDXUTSDKMesh                g_OpaqueMesh;
CDXUTSDKMesh                g_TransparentMesh[NUM_TRANSPARENT_MESHES];
int                         g_nCurrentModel = NUM_TRANSPARENT_MESHES-1;
DWORD                       g_uRandSeed = 0;
ID3D11InputLayout*          g_pSimpleVertexLayout = NULL;
ID3D11InputLayout*          g_pExtendedVertexLayout = NULL;
OBJECT_STRUCT               g_pCubeList[MAX_NUMBER_OF_CUBES];
ID3D11Buffer*               g_pCubeVB = NULL;
ID3D11Buffer*               g_pCubeIB = NULL;
ID3D11Buffer*               g_pFullscreenQuadVB = NULL;
RECT                        g_ModelBoundingBoxExtent;

// Constant buffers
ID3D11Buffer*               g_pMainCB = NULL;
ID3D11Buffer*               g_pCubeCB[MAX_NUMBER_OF_CUBES];
ID3D11Buffer*               g_pOpaqueMeshCB;
ID3D11Buffer*               g_pTransparentMeshCB;
ID3D11Buffer*               g_pTileCoordinatesCB;

// Shaders
ID3D11VertexShader*         g_pVSPassThrough = NULL;
ID3D11VertexShader*         g_pMainVS = NULL;
ID3D11PixelShader*          g_pLightingAndTexturingPS = NULL;
ID3D11PixelShader*          g_pStoreFragmentsPS_LinkedList = NULL;
ID3D11PixelShader*          g_pRenderFragmentsPS_LinkedList = NULL;
ID3D11PixelShader*          g_pRenderFragmentsWithResolvePS_LinkedList = NULL;


// Textures
ID3D11ShaderResourceView*   g_pDiffuseTextureRV = NULL;
ID3D11ShaderResourceView*   g_pWhiteTextureRV = NULL;

// Depth stencil buffer variables
ID3D11Texture2D*            g_pDepthStencilTexture = NULL;
ID3D11DepthStencilView*     g_pDepthStencilTextureDSV = NULL;

// Start offset buffer
ID3D11Buffer*               g_pStartOffsetBuffer = NULL;
ID3D11UnorderedAccessView*  g_pStartOffsetBufferUAV = NULL;
ID3D11ShaderResourceView*   g_pStartOffsetBufferSRV = NULL;

// LINKED LIST resources
ID3D11Buffer*               g_pFragmentAndLinkStructuredBuffer = NULL;
ID3D11UnorderedAccessView*  g_pFragmentAndLinkStructuredBufferUAV = NULL;
ID3D11ShaderResourceView*   g_pFragmentAndLinkStructuredBufferSRV = NULL;

// Main render target
ID3D11Texture2D*            g_pMainRenderTargetTexture = NULL;
ID3D11ShaderResourceView*   g_pMainRenderTargetTextureSRV = NULL;
ID3D11RenderTargetView*     g_pMainRenderTargetTextureRTV = NULL;
ID3D11Texture2D*            g_pMainRenderTargetTextureResolved = NULL;
ID3D11ShaderResourceView*   g_pMainRenderTargetTextureResolvedSRV = NULL;
ID3D11RenderTargetView*     g_pMainRenderTargetTextureResolvedRTV = NULL;
ID3D11UnorderedAccessView*  g_pMainRenderTargetTextureResolvedUAV = NULL;

// Copy of back buffer texture
ID3D11Texture2D*            g_pCopyOfMainRenderTargetTexture = NULL;
ID3D11ShaderResourceView*   g_pCopyOfMainRenderTargetTextureSRV = NULL;
ID3D11RenderTargetView*     g_pCopyOfMainRenderTargetTextureRTV = NULL;

// State objects
ID3D11RasterizerState*      g_pRasterizerStateSolid = NULL;
ID3D11RasterizerState*      g_pRasterizerStateWireframe = NULL;
ID3D11SamplerState*         g_pSamplerStateLinearWrap = NULL;
ID3D11SamplerState*         g_pSamplerStateLinearClamp = NULL;
ID3D11SamplerState*         g_pSamplerStatePointClamp = NULL;
ID3D11BlendState*           g_pBlendStateNoBlend = NULL;
ID3D11BlendState*           g_pBlendStateSrcAlphaInvSrcAlphaBlend = NULL;
//ID3D11BlendState*            g_pBlendStateBackgroundUnderBlending = NULL;
ID3D11BlendState*           g_pColorWritesOff = NULL;
ID3D11DepthStencilState*    g_pDepthTestDisabledDSS = NULL;
ID3D11DepthStencilState*    g_pDepthTestEnabledDSS = NULL;
ID3D11DepthStencilState*    g_pDepthTestEnabledNoDepthWritesDSS = NULL;
ID3D11DepthStencilState*    g_pDepthTestEnabledNoDepthWritesStencilWriteIncrementDSS = NULL;
ID3D11DepthStencilState*    g_pDepthTestDisabledStencilTestLessDSS = NULL;

// Query
ID3D11Query*                g_pEventQuery = NULL;

// Camera parameters
D3DXVECTOR3                 g_vecEye(0.0f, 10.0f, -25.0f);
D3DXVECTOR3                 g_vecAt (0.0f, 0.0f, 0.0f);

// Mouse
bool                        g_bLeftButtonDown = false;
bool                        g_bRightButtonDown = false;
bool                        g_bMiddleButtonDown = false;

// Render settings
int                         g_nOITMethod = OITMETHOD_LINKEDLIST;
int                         g_nPreviousOITMethod = OITMETHOD_LINKEDLIST; 
float                       g_fScreenWidth;
float                       g_fScreenHeight;
int                         g_nRenderHUD = 2;
BOOL                        g_bEnableWireFrame = false;
int                         g_nNumCubes = 0;
bool                        g_bMSAAResolveDuringSort = FALSE;
DXGI_SAMPLE_DESC            g_MSAASampleDesc = { 1, 0 };
DXGI_SAMPLE_DESC            g_PreviousMSAASampleDesc = { 1, 0 };
bool                        g_bLinkedListUseComputeShader = FALSE;
bool                        g_bRecreateResources = TRUE;
DXGI_SURFACE_DESC           g_pBackBufferSurfaceDesc;

// Debug variables
ID3D11Texture2D*            g_pCopyOfMainRenderTargetTextureSTAGING = NULL;
DWORD                       g_dwFrameNumberToDump = 0;    // 0 means never dump to disk (frame counter starts at 1)
int                         g_nNumTilesVisible = 0;


//--------------------------------------------------------------------------------------
// UI control IDs
//--------------------------------------------------------------------------------------
#define IDC_TOGGLEFULLSCREEN                    1
#define IDC_TOGGLEREF                           3
#define IDC_CHANGEDEVICE                        4
#define IDC_STATIC_OIT                          10
#define IDC_RADIOBUTTON_OIT_DISABLED            11
#define IDC_RADIOBUTTON_OIT_LINKEDLIST          12
#define IDC_STATIC_MODEL_SELECTION              20
#define IDC_COMBOBOX_MODEL_SELECTION            21
#define IDC_STATIC_NUMBER_OF_CUBES              30
#define IDC_SLIDER_NUMBER_OF_CUBES              31
#define IDC_CHECKBOX_LINKED_LIST_MSAA_RESOLVE   44
#define IDC_STATIC_MSAA                         50
#define IDC_RADIOBUTTON_MSAA_1X                 51                          
#define IDC_RADIOBUTTON_MSAA_2X                 52
#define IDC_RADIOBUTTON_MSAA_4X                 53

//--------------------------------------------------------------------------------------
// Forward declarations 
//--------------------------------------------------------------------------------------
bool CALLBACK ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, void* pUserContext );
void CALLBACK OnFrameMove( double fTime, float fElapsedTime, void* pUserContext );
LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing,
                          void* pUserContext );
void    CALLBACK MouseProc( bool bLeftButtonDown, bool bRightButtonDown, bool bMiddleButtonDown, 
                            bool bSideButton1Down, bool bSideButton2Down, int nMouseWheelDelta, 
                            int xPos, int yPos, void* pUserContext );
void CALLBACK OnKeyboard( UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext );
void CALLBACK OnGUIEvent( UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext );
bool CALLBACK IsD3D11DeviceAcceptable( const CD3D11EnumAdapterInfo *AdapterInfo, UINT Output, const CD3D11EnumDeviceInfo *DeviceInfo,
                                      DXGI_FORMAT BackBufferFormat, bool bWindowed, void* pUserContext );
HRESULT CALLBACK OnD3D11CreateDevice( ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc,
                                      void* pUserContext );
HRESULT CALLBACK OnD3D11ResizedSwapChain( ID3D11Device* pd3dDevice, IDXGISwapChain* pSwapChain,
                                          const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext );
void CALLBACK OnD3D11ReleasingSwapChain( void* pUserContext );
void CALLBACK OnD3D11DestroyDevice( void* pUserContext );
void CALLBACK OnD3D11FrameRender( ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext, double fTime,
                                  float fElapsedTime, void* pUserContext );
HRESULT CreateResources(ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc);
void DestroyResources();
void InitApp();
void RenderText();
void InitializeCubeList();
void UpdateCubeConstantBuffers(ID3D11DeviceContext* pd3dContext);
void RenderCubes(ID3D11DeviceContext* pd3dContext);
void RenderModel(ID3D11DeviceContext* pd3dContext, CDXUTSDKMesh* pDXUTMesh, bool bUseBoundingBoxTesting=false);
bool IsNextArg( WCHAR*& strCmdLine, WCHAR* strArg );
bool GetCmdParam( WCHAR*& strCmdLine, WCHAR* strFlag );
void CreateStagingFromTexture2D(ID3D11Device *pDev, ID3D11Texture2D* pTex, ID3D11Texture2D** ppTexSTAGING);
void CreateStagingFromBuffer(ID3D11Device *pDev, ID3D11Buffer* pBuffer, ID3D11Buffer** ppBufferSTAGING);
void SaveScene();
bool LoadScene();
void CalculateOffCenteredProjectionMatrixFrom2DRectangle(D3DXMATRIX* pOffCenteredProjectionMatrix, 
                                                         float fFullscreenWidth, float fFullscreenHeight, float fFulscreenZNear, float fFullscreenZFar, float fFullscreenVerticalFOV, 
                                                         RECT RectArea);
void PerformBoundingBoxCheck(CDXUTSDKMesh* pSDKMesh, D3DXMATRIX* pTransformationMatrix);
void GetTransformedBoundingBoxExtents(CDXUTSDKMesh* pSDKMesh, D3DXMATRIX* pTransformationMatrix, float fRenderWidth, float fRenderHeight, 
                                      RECT *pBoundingBoxExtent);
void NormalizePlane( D3DXVECTOR4* pPlaneEquation );
void ExtractPlanesFromFrustum( D3DXVECTOR4* pPlaneEquation, const D3DXMATRIX* pMatrix, bool bNormalize=TRUE);
float DistanceToPoint( const D3DXVECTOR4 *pPlaneEquation, const D3DXVECTOR3* pPoint );
float AxisAlignedBoundingBoxWithinFrustum( BOUNDING_BOX_STRUCT* pBoundingBox, D3DXVECTOR4* pFrustumPlaneEquations );


//--------------------------------------------------------------------------------------
// Entry point to the program. Initializes everything and goes into a message processing 
// loop. Idle time is used to render the scene.
//--------------------------------------------------------------------------------------
int WINAPI wWinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow )
{
    // Enable run-time memory check for debug builds.
#if defined(_DEBUG)
    _CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
#endif

    // DXUT will create and use the best device (either D3D10 or D3D11) 
    // that is available on the system depending on which D3D callbacks are set below

    // Set DXUT callbacks
    DXUTSetCallbackDeviceChanging( ModifyDeviceSettings );
    DXUTSetCallbackMsgProc( MsgProc );
    DXUTSetCallbackFrameMove( OnFrameMove );
    DXUTSetCallbackMouse( MouseProc );

    DXUTSetCallbackD3D11DeviceAcceptable( IsD3D11DeviceAcceptable, NULL );
    DXUTSetCallbackD3D11DeviceCreated( OnD3D11CreateDevice, NULL );
    DXUTSetCallbackD3D11SwapChainResized( OnD3D11ResizedSwapChain );
    DXUTSetCallbackD3D11FrameRender( OnD3D11FrameRender );
    DXUTSetCallbackD3D11SwapChainReleasing( OnD3D11ReleasingSwapChain );
    DXUTSetCallbackD3D11DeviceDestroyed( OnD3D11DestroyDevice );

    DXUTSetCallbackKeyboard( OnKeyboard );

        // Perform application-dependant command line processing
    WCHAR* strCmdLine = GetCommandLine();
    WCHAR strFlag[MAX_PATH];
    int nNumArgs;
    WCHAR** pstrArgList = CommandLineToArgvW( strCmdLine, &nNumArgs );
    for( int iArg=1; iArg<nNumArgs; iArg++ )
    {
        strCmdLine = pstrArgList[iArg];

        // Handle flag args
        if( *strCmdLine == L'/' || *strCmdLine == L'-' )
        {
            strCmdLine++;

            if( IsNextArg( strCmdLine, L"numcubes" ) )
            {
                if( GetCmdParam( strCmdLine, strFlag ) )
                {
                   g_nNumCubes = (int)_wtoi(strFlag);
                }
                continue;
            }

            if( IsNextArg( strCmdLine, L"dumpframe" ) )
            {
                if( GetCmdParam( strCmdLine, strFlag ) )
                {
                   g_dwFrameNumberToDump = _wtoi(strFlag);
                }
                continue;
            }
            
            if( IsNextArg( strCmdLine, L"method" ) )
             {
                if( GetCmdParam( strCmdLine, strFlag ) )
                {
                   if ( (wcscmp(strFlag, L"nosorting")==0) )
                   {
                      g_nOITMethod = OITMETHOD_NOSORTING;
                   }
                   else if ( (wcscmp(strFlag, L"linkedlist")==0) )
                   {
                      g_nOITMethod = OITMETHOD_LINKEDLIST;
                   }
                }
            }

            if( IsNextArg( strCmdLine, L"wireframe" ) )
            {
                g_bEnableWireFrame = true;
                continue;
            }

            if( IsNextArg( strCmdLine, L"notext" ) )
            {
                g_nRenderHUD = 0;
                continue;
            }
        }
    }

    InitApp();
    DXUTInit( true, true );
    DXUTSetCursorSettings( true, true ); // Show the cursor and clip it when in full screen
    DXUTCreateWindow( L"OIT11LinkedLists v1.0" );
    DXUTCreateDevice( D3D_FEATURE_LEVEL_11_0, true, 1024, 768);
    DXUTMainLoop(); // Enter into the DXUT render loop

    return DXUTGetExitCode();
}


//--------------------------------------------------------------------------------------
// Handle mouse buttons
//--------------------------------------------------------------------------------------
void CALLBACK MouseProc( bool bLeftButtonDown, bool bRightButtonDown, 
                         bool bMiddleButtonDown, bool bSideButton1Down, 
                         bool bSideButton2Down, int nMouseWheelDelta, 
                         int xPos, int yPos, void* pUserContext )
{
    bool bOldLeftButtonDown = g_bLeftButtonDown;
    bool bOldRightButtonDown = g_bRightButtonDown;
    bool bOldMiddleButtonDown = g_bMiddleButtonDown;
    g_bLeftButtonDown = bLeftButtonDown;
    g_bMiddleButtonDown = bMiddleButtonDown;
    g_bRightButtonDown = bRightButtonDown;

    if( bOldLeftButtonDown && !g_bLeftButtonDown )
    {
        g_Camera.SetEnablePositionMovement( false );
    }
    else if( !bOldLeftButtonDown && g_bLeftButtonDown )
    {
        g_Camera.SetEnablePositionMovement( true );
    }

    if( !bOldRightButtonDown && g_bRightButtonDown )
    {
        g_Camera.SetEnablePositionMovement( false );
    }

    if( bOldMiddleButtonDown && !g_bMiddleButtonDown )
    {
        g_LCamera.SetEnablePositionMovement( false );
    } 
    else if( !bOldMiddleButtonDown && g_bMiddleButtonDown )
    {
        g_LCamera.SetEnablePositionMovement( true );
        g_Camera.SetEnablePositionMovement( false );
    }

    // If no mouse button is down at all, enable camera movement.
    if( !g_bLeftButtonDown && !g_bRightButtonDown && !g_bMiddleButtonDown )
    {
        g_Camera.SetEnablePositionMovement( true );
    }
}

//--------------------------------------------------------------------------------------
// Initialize the app 
//--------------------------------------------------------------------------------------
void InitApp()
{
    // Initialize dialogs
    g_D3DSettingsDlg.Init( &g_DialogResourceManager );
    g_HUD.Init( &g_DialogResourceManager );
    g_SampleUI.Init( &g_DialogResourceManager );

    g_HUD.SetCallback( OnGUIEvent ); int iY = 10;
    g_HUD.AddButton( IDC_TOGGLEFULLSCREEN, L"Toggle full screen", 35, iY, 125, 22 );
    g_HUD.AddButton( IDC_TOGGLEREF, L"Toggle REF (F3)", 35, iY += 24, 125, 22, VK_F3 );
    g_HUD.AddButton( IDC_CHANGEDEVICE, L"Change device (F2)", 35, iY += 24, 125, 22, VK_F2 );

    iY = 0;
    int iX = 15;
    g_SampleUI.AddStatic( IDC_STATIC_MSAA, L"MSAA:", iX, iY += 20, 55, 24 );
    g_SampleUI.AddRadioButton( IDC_RADIOBUTTON_MSAA_1X, 9, L"Off", iX, iY+=20, 
                               40, 18, g_MSAASampleDesc.Count==1 ? true : false);
    g_SampleUI.AddRadioButton( IDC_RADIOBUTTON_MSAA_2X, 9, L"2x", iX+50, iY, 
                               40, 18, g_MSAASampleDesc.Count==2 ? true : false);
    g_SampleUI.AddRadioButton( IDC_RADIOBUTTON_MSAA_4X, 9, L"4x", iX+100, iY, 
                               40, 18, g_MSAASampleDesc.Count==4 ? true : false);

    g_SampleUI.AddStatic( IDC_STATIC_OIT, L"OIT Method:", iX, iY += 30, 55, 24 );
    g_SampleUI.AddRadioButton( IDC_RADIOBUTTON_OIT_DISABLED, 0, L"DISABLED", iX, iY += 25, 
                               55, 24, g_nOITMethod==OITMETHOD_NOSORTING ? true : false, '1');
    g_SampleUI.AddRadioButton( IDC_RADIOBUTTON_OIT_LINKEDLIST, 0, L"LINKED LISTS", iX, iY += 25, 
                               55, 24, g_nOITMethod==OITMETHOD_LINKEDLIST ? true : false, '2');

    g_SampleUI.AddCheckBox(IDC_CHECKBOX_LINKED_LIST_MSAA_RESOLVE, L"Fragments resolve", iX+5, iY+=25, 
                           100, 18, g_bMSAAResolveDuringSort);
    
    g_SampleUI.GetControl( IDC_CHECKBOX_LINKED_LIST_MSAA_RESOLVE )->SetEnabled( g_nOITMethod==OITMETHOD_LINKEDLIST && g_MSAASampleDesc.Count>1 );
    g_SampleUI.GetControl( IDC_CHECKBOX_LINKED_LIST_MSAA_RESOLVE )->SetVisible( g_nOITMethod==OITMETHOD_LINKEDLIST && g_MSAASampleDesc.Count>1 );
        
    g_SampleUI.AddStatic( IDC_STATIC_MODEL_SELECTION, L"(M)odel:", iX, iY += 25, 55, 24 );
    CDXUTComboBox *pCombo;
    g_SampleUI.AddComboBox( IDC_COMBOBOX_MODEL_SELECTION, iX, iY += 25, 140, 24, 'M', false, &pCombo );
    if( pCombo )
    {
        pCombo->SetDropHeight( 25 );
        for (int i=0; i<NUM_TRANSPARENT_MESHES; ++i)
        {
            pCombo->AddItem( vTransparentMeshesInScene[i].sMeshName, NULL );
        }

        pCombo->SetSelectedByIndex( g_nCurrentModel );
    }

    g_SampleUI.AddStatic( IDC_STATIC_NUMBER_OF_CUBES, L"Opaque cubes:", iX+5, iY+=25, 108, 24 );
    g_SampleUI.AddSlider( IDC_SLIDER_NUMBER_OF_CUBES, iX, iY += 25, 140, 24, 0, 
                          (int)(MAX_NUMBER_OF_CUBES), g_nNumCubes, false );

    g_SampleUI.SetCallback( OnGUIEvent ); iY = 10;

    // Setup the camera's view parameters
    g_Camera.SetRotateButtons( true, false, false );
    g_Camera.SetEnablePositionMovement(true);
    g_Camera.SetViewParams( &g_vecEye, &g_vecAt );
    g_Camera.SetScalers(0.0025f, 5.0f);
    
    D3DXVECTOR3 vecLight( 0,0,0 );
    D3DXVECTOR3 vecLightAt ( 0.0f, -0.5f, -1.0f );
    g_LCamera.SetViewParams( &vecLight, &vecLightAt );
    g_LCamera.SetButtonMasks( MOUSE_RIGHT_BUTTON, 0, 0 );

    // Load default scene
    if (!LoadScene())
    {
        // If scene file doesn't exist then create new one
        InitializeCubeList();
    }
}


//--------------------------------------------------------------------------------------
// Called right before creating a D3D9 or D3D10 device, allowing the app to modify the device settings as needed
//--------------------------------------------------------------------------------------
bool CALLBACK ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, void* pUserContext )
{
    // Do not automatically create a depth stencil buffer; we'll do this ourselves
    pDeviceSettings->d3d11.AutoCreateDepthStencil = false;

    // For the first device created if its a REF device, optionally display a warning dialog box
    static bool s_bFirstTime = true;
    if( s_bFirstTime )
    {
        s_bFirstTime = false;
        if( ( DXUT_D3D11_DEVICE == pDeviceSettings->ver &&
              pDeviceSettings->d3d11.DriverType == D3D_DRIVER_TYPE_REFERENCE ) )
        {
            DXUTDisplaySwitchingToREFWarning( pDeviceSettings->ver );
        }

        // Disable vsync
        pDeviceSettings->d3d11.SyncInterval = 0;
        g_D3DSettingsDlg.GetDialogControl()->GetComboBox( DXUTSETTINGSDLG_PRESENT_INTERVAL )->SetEnabled( false );
    }

    // MSAA is disabled on the back buffer regardless of MSAA settings; this is because 
    // all MSAA will be done on the main render target
    DXGI_SAMPLE_DESC NoMSAAASampleDesc = { 1, 0 };
    pDeviceSettings->d3d11.sd.SampleDesc = NoMSAAASampleDesc;

    return true;
}


//--------------------------------------------------------------------------------------
// Handle updates to the scene
//--------------------------------------------------------------------------------------
void CALLBACK OnFrameMove( double fTime, float fElapsedTime, void* pUserContext )
{
    // Update the camera's position based on user input 
    g_Camera.FrameMove( fElapsedTime );
    g_LCamera.FrameMove( fElapsedTime );
}


//--------------------------------------------------------------------------------------
// Render the help and statistics text
//--------------------------------------------------------------------------------------
void RenderText()
{
    g_pTxtHelper->Begin();
    g_pTxtHelper->SetInsertionPos( 2, 0 );
    g_pTxtHelper->SetForegroundColor( D3DXCOLOR( 1.0f, 1.0f, 0.0f, 1.0f ) );
    g_pTxtHelper->DrawTextLine( DXUTGetFrameStats( DXUTIsVsyncEnabled() ) );
    g_pTxtHelper->DrawTextLine( DXUTGetDeviceStats() );

#if (NUMBER_OF_TILES_X!=1 || NUMBER_OF_TILES_Y!=1)
    if (g_nOITMethod==OITMETHOD_LINKEDLIST)
    {
        WCHAR pszTextString[128];
        swprintf_s( pszTextString, sizeof(pszTextString)/sizeof(*pszTextString), L"Model bounding box 2D extents: (%d, %d) -> (%d, %d)", 
            g_ModelBoundingBoxExtent.left, g_ModelBoundingBoxExtent.top, g_ModelBoundingBoxExtent.right, g_ModelBoundingBoxExtent.bottom);
        g_pTxtHelper->DrawTextLine( pszTextString );
        swprintf_s( pszTextString, sizeof(pszTextString)/sizeof(*pszTextString), L"Tiles visible:%d / %d", 
                    g_nNumTilesVisible, NUMBER_OF_TILES_X * NUMBER_OF_TILES_Y);
        g_pTxtHelper->DrawTextLine( pszTextString );
    }
#endif
 
    g_pTxtHelper->End();
}


//--------------------------------------------------------------------------------------
// Handle messages to the application
//--------------------------------------------------------------------------------------
LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing,
                          void* pUserContext )
{
    // Pass messages to dialog resource manager calls so GUI state is updated correctly
    *pbNoFurtherProcessing = g_DialogResourceManager.MsgProc( hWnd, uMsg, wParam, lParam );
    if( *pbNoFurtherProcessing )
        return 0;

    // Pass messages to settings dialog if its active
    if( g_D3DSettingsDlg.IsActive() )
    {
        g_D3DSettingsDlg.MsgProc( hWnd, uMsg, wParam, lParam );
        return 0;
    }

    // Give the dialogs a chance to handle the message first
    *pbNoFurtherProcessing = g_HUD.MsgProc( hWnd, uMsg, wParam, lParam );
    if( *pbNoFurtherProcessing )
        return 0;
    *pbNoFurtherProcessing = g_SampleUI.MsgProc( hWnd, uMsg, wParam, lParam );
    if( *pbNoFurtherProcessing )
        return 0;

    // Pass all remaining windows messages to camera so it can respond to user input
    g_Camera.HandleMessages( hWnd, uMsg, wParam, lParam );
    g_LCamera.HandleMessages( hWnd, uMsg, wParam, lParam );

    return 0;
}


//--------------------------------------------------------------------------------------
// Handle key presses
//--------------------------------------------------------------------------------------
void CALLBACK OnKeyboard( UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext )
{
    if( bKeyDown )
    {
        switch( nChar )
        {
            case VK_F6:         SaveScene(); break;
            case VK_F9:         LoadScene(); 
                                break;
            case 'R':           g_uRandSeed = (DWORD)rand(); 
                                InitializeCubeList();
                                break;
            case 'F':           g_bEnableWireFrame = !g_bEnableWireFrame; break;
            case 'H':           g_nRenderHUD = (g_nRenderHUD+1)%3; break;
        }
    }
}

//--------------------------------------------------------------------------------------
// Handles the GUI events
//--------------------------------------------------------------------------------------
void CALLBACK OnGUIEvent( UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext )
{
    switch( nControlID )
    {
        // Standard DXUT controls
        case IDC_TOGGLEFULLSCREEN:
            DXUTToggleFullScreen(); break;
        case IDC_TOGGLEREF:
            DXUTToggleREF(); break;
        case IDC_CHANGEDEVICE:
            g_D3DSettingsDlg.SetActive( !g_D3DSettingsDlg.IsActive() ); break;

        // MSAA mode
        case IDC_RADIOBUTTON_MSAA_1X: 
        {
            g_MSAASampleDesc.Count = 1;
            if (g_PreviousMSAASampleDesc.Count != g_MSAASampleDesc.Count)
            {
                g_bRecreateResources=TRUE;
                g_PreviousMSAASampleDesc.Count = g_MSAASampleDesc.Count;
            }

            if (g_nOITMethod==OITMETHOD_LINKEDLIST)
            {
                g_SampleUI.GetControl( IDC_CHECKBOX_LINKED_LIST_MSAA_RESOLVE )->SetEnabled( false );
                g_SampleUI.GetControl( IDC_CHECKBOX_LINKED_LIST_MSAA_RESOLVE )->SetVisible( false );
            }
            break;
        }
        case IDC_RADIOBUTTON_MSAA_2X: 
        {
            g_MSAASampleDesc.Count = 2;
            if (g_PreviousMSAASampleDesc.Count != g_MSAASampleDesc.Count)
            {
                g_bRecreateResources=TRUE;
                g_PreviousMSAASampleDesc.Count = g_MSAASampleDesc.Count;
            }

            if (g_nOITMethod==OITMETHOD_LINKEDLIST)
            {
                g_SampleUI.GetControl( IDC_CHECKBOX_LINKED_LIST_MSAA_RESOLVE )->SetEnabled( true );
                g_SampleUI.GetControl( IDC_CHECKBOX_LINKED_LIST_MSAA_RESOLVE )->SetVisible( true );
            }
            break;
        }
        case IDC_RADIOBUTTON_MSAA_4X: 
        {
            g_MSAASampleDesc.Count = 4;
            if (g_PreviousMSAASampleDesc.Count != g_MSAASampleDesc.Count)
            {
                g_bRecreateResources=TRUE;
                g_PreviousMSAASampleDesc.Count = g_MSAASampleDesc.Count;
            }

            if (g_nOITMethod==OITMETHOD_LINKEDLIST)
            {
                g_SampleUI.GetControl( IDC_CHECKBOX_LINKED_LIST_MSAA_RESOLVE )->SetEnabled( true );
                g_SampleUI.GetControl( IDC_CHECKBOX_LINKED_LIST_MSAA_RESOLVE )->SetVisible( true );
            }
            break;
        }
            
        // OIT method
        case IDC_RADIOBUTTON_OIT_DISABLED:          
        {
            g_nOITMethod = OITMETHOD_NOSORTING;
            if (g_nPreviousOITMethod!=g_nOITMethod)
            {
                g_bRecreateResources=TRUE;
                g_nPreviousOITMethod = g_nOITMethod;
            }
            g_SampleUI.GetControl( IDC_CHECKBOX_LINKED_LIST_MSAA_RESOLVE )->SetEnabled( false );
            g_SampleUI.GetControl( IDC_CHECKBOX_LINKED_LIST_MSAA_RESOLVE )->SetVisible( false );
            break;
        }
        case IDC_RADIOBUTTON_OIT_LINKEDLIST:        
        {
            g_nOITMethod = OITMETHOD_LINKEDLIST;
            if (g_nPreviousOITMethod!=g_nOITMethod)
            {
                g_bRecreateResources=TRUE;
                g_nPreviousOITMethod = g_nOITMethod;
            }
            g_SampleUI.GetControl( IDC_CHECKBOX_LINKED_LIST_MSAA_RESOLVE )->SetEnabled( g_MSAASampleDesc.Count>1 );
            g_SampleUI.GetControl( IDC_CHECKBOX_LINKED_LIST_MSAA_RESOLVE )->SetVisible( g_MSAASampleDesc.Count>1 );
            break;
        }

        // Linked-list related controls
        case IDC_CHECKBOX_LINKED_LIST_MSAA_RESOLVE:
        {
            g_bMSAAResolveDuringSort = ((CDXUTCheckBox*)pControl)->GetChecked(); 
            break;
        }

        // Model selection
        case IDC_COMBOBOX_MODEL_SELECTION:
        {
            g_nCurrentModel = ((CDXUTComboBox*)pControl)->GetSelectedIndex(); break;
        }

        // Number of cubes
        case IDC_SLIDER_NUMBER_OF_CUBES:
        {
            g_nNumCubes = ((CDXUTSlider*)pControl)->GetValue();
        }
        break;
    }
}

//--------------------------------------------------------------------------------------
// Reject any D3D11 devices that aren't acceptable by returning false
//--------------------------------------------------------------------------------------
bool CALLBACK IsD3D11DeviceAcceptable( const CD3D11EnumAdapterInfo *AdapterInfo, UINT Output, 
                                       const CD3D11EnumDeviceInfo *DeviceInfo, DXGI_FORMAT BackBufferFormat, 
                                       bool bWindowed, void* pUserContext )
{
    return true;
}



//--------------------------------------------------------------------------------------
// Create any D3D10 resources that aren't dependant on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D11CreateDevice( ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc,
                                      void* pUserContext )
{
    HRESULT                hr;
    D3D11_BUFFER_DESC   bd;
    ID3DBlob*            pErrorBlob = NULL;

    ID3D11DeviceContext* pd3dImmediateContext = DXUTGetD3D11DeviceContext();
    V_RETURN( g_DialogResourceManager.OnD3D11CreateDevice( pd3dDevice, pd3dImmediateContext ) );
    V_RETURN( g_D3DSettingsDlg.OnD3D11CreateDevice( pd3dDevice ) );
    g_pTxtHelper = new CDXUTTextHelper( pd3dDevice, pd3dImmediateContext, &g_DialogResourceManager, 15 );

    //
    // Shaders
    //

    DWORD dwShaderFlags = D3D10_SHADER_ENABLE_STRICTNESS;
#if defined( DEBUG ) || defined( _DEBUG )
    // Set the D3D10_SHADER_DEBUG flag to embed debug information in the shaders.
    // Setting this flag improves the shader debugging experience, but still allows 
    // the shaders to be optimized and to run exactly the way they will run in 
    // the release conf2iguration of this program.
    dwShaderFlags |= D3D10_SHADER_DEBUG;
#endif

    ID3DBlob* pBlobVSPassThrough = NULL;
    ID3DBlob* pBlobVS = NULL;
    ID3DBlob* pBlobPS_LightingAndTexturing = NULL;

    // Generic shaders
    hr = D3DX11CompileFromFile( L"OIT11LinkedLists.hlsl", NULL, NULL, "VSPassThrough", "vs_5_0", dwShaderFlags, 
                                0, NULL, &pBlobVSPassThrough, &pErrorBlob, NULL);
    hr = D3DX11CompileFromFile( L"OIT11LinkedLists.hlsl", NULL, NULL, "VS", "vs_5_0", dwShaderFlags, 
                                0, NULL, &pBlobVS, &pErrorBlob, NULL);
    hr = D3DX11CompileFromFile( L"OIT11LinkedLists.hlsl", NULL, NULL, "PS_LightingAndTexturing", "ps_5_0", dwShaderFlags, 
                                0, NULL, &pBlobPS_LightingAndTexturing, &pErrorBlob, NULL);
    V_RETURN( pd3dDevice->CreateVertexShader( pBlobVSPassThrough->GetBufferPointer(), pBlobVSPassThrough->GetBufferSize(), 
                                              NULL, &g_pVSPassThrough ) );
    V_RETURN( pd3dDevice->CreateVertexShader( pBlobVS->GetBufferPointer(), pBlobVS->GetBufferSize(), 
                                              NULL, &g_pMainVS ) );
    V_RETURN( pd3dDevice->CreatePixelShader( pBlobPS_LightingAndTexturing->GetBufferPointer(), pBlobPS_LightingAndTexturing->GetBufferSize(), 
                                              NULL, &g_pLightingAndTexturingPS ) );
    
    // 
    // Meshes
    //

    // Load opaque mesh
    hr = g_OpaqueMesh.Create( pd3dDevice, L"misc\\cell.sdkmesh" );
    assert( D3D_OK == hr );

    // Load all transparent meshes
    for (int i=0; i<NUM_TRANSPARENT_MESHES; i++)
    {
        hr = g_TransparentMesh[i].Create( pd3dDevice, vTransparentMeshesInScene[i].sMeshFileName );
        assert( D3D_OK == hr );
    }

    // Cube vertex buffer
    EXTENDEDVERTEX CubeVertices[] =
    {
        { -1.0f,  1.0f, -1.0f , -0.577f,  0.577f, -0.577f , 0.0f, 1.0f  },
        {  1.0f,  1.0f, -1.0f ,  0.577f,  0.577f, -0.577f , 1.0f, 1.0f  },
        {  1.0f,  1.0f, 1.0f ,   0.577f,  0.577f,  0.577f , 1.0f, 0.0f  },
        { -1.0f,  1.0f, 1.0f ,  -0.577f,  0.577f,  0.577f , 0.0f, 0.0f  },

        { -1.0f, -1.0f, -1.0f , -0.577f, -0.577f, -0.577f , 0.0f, 1.0f  },
        {  1.0f, -1.0f, -1.0f ,  0.577f, -0.577f, -0.577f , 1.0f, 1.0f  },
        {  1.0f, -1.0f, 1.0f ,   0.577f, -0.577f,  0.577f , 1.0f, 0.0f  },
        { -1.0f, -1.0f, 1.0f ,  -0.577f, -0.577f,  0.577f , 0.0f, 0.0f  },

        { -1.0f, -1.0f, 1.0f ,  -0.577f, -0.577f,  0.577f , 0.0f, 1.0f  },
        { -1.0f, -1.0f, -1.0f , -0.577f, -0.577f, -0.577f , 1.0f, 1.0f  },
        { -1.0f,  1.0f, -1.0f , -0.577f,  0.577f, -0.577f , 1.0f, 0.0f  },
        { -1.0f,  1.0f, 1.0f ,  -0.577f,  0.577f,  0.577f , 0.0f, 0.0f  },

        {  1.0f, -1.0f, 1.0f ,   0.577f, -0.577f,  0.577f , 0.0f, 1.0f  },
        {  1.0f, -1.0f, -1.0f ,  0.577f, -0.577f, -0.577f , 1.0f, 1.0f  },
        {  1.0f,  1.0f, -1.0f ,  0.577f,  0.577f, -0.577f , 1.0f, 0.0f  },
        {  1.0f,  1.0f, 1.0f ,   0.577f,  0.577f,  0.577f , 0.0f, 0.0f  },

        { -1.0f, -1.0f, -1.0f , -0.577f, -0.577f, -0.577f , 0.0f, 1.0f  },
        {  1.0f, -1.0f, -1.0f ,  0.577f, -0.577f, -0.577f , 1.0f, 1.0f  },
        {  1.0f,  1.0f, -1.0f ,  0.577f,  0.577f, -0.577f , 1.0f, 0.0f  },
        { -1.0f,  1.0f, -1.0f , -0.577f,  0.577f, -0.577f , 0.0f, 0.0f  },

        { -1.0f, -1.0f, 1.0f ,  -0.577f, -0.577f,  0.577f , 0.0f, 1.0f  },
        {  1.0f, -1.0f, 1.0f ,   0.577f, -0.577f,  0.577f , 1.0f, 1.0f  },
        {  1.0f,  1.0f, 1.0f ,   0.577f,  0.577f,  0.577f , 1.0f, 0.0f  },
        { -1.0f,  1.0f, 1.0f ,  -0.577f,  0.577f,  0.577f , 0.0f, 0.0f  },
    };

    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.ByteWidth = sizeof( EXTENDEDVERTEX ) * 24;
    bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    bd.CPUAccessFlags = 0;
    bd.MiscFlags = 0;
    D3D11_SUBRESOURCE_DATA InitData;
    InitData.pSysMem = CubeVertices;
    hr = pd3dDevice->CreateBuffer( &bd, &InitData, &g_pCubeVB );
    if( FAILED( hr ) )
        return hr;

    // Cube index buffer
    WORD CubeIndices[] =
    {
        3,1,0,
        2,1,3,

        6,4,5,
        7,4,6,

        11,9,8,
        10,9,11,

        14,12,13,
        15,12,14,

        19,17,16,
        18,17,19,

        22,20,21,
        23,20,22
    };
    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.ByteWidth = sizeof( WORD ) * 36;
    bd.BindFlags = D3D11_BIND_INDEX_BUFFER;
    bd.CPUAccessFlags = 0;
    bd.MiscFlags = 0;
    InitData.pSysMem = CubeIndices;
    hr = pd3dDevice->CreateBuffer( &bd, &InitData, &g_pCubeIB );
    if( FAILED( hr ) )
        return hr;

    // Create fullscreen quad geometry
    InitData.pSysMem = g_FullscreenQuad;
    bd.Usage =          D3D11_USAGE_DEFAULT;
    bd.ByteWidth =      sizeof( SIMPLEVERTEX ) * 4;
    bd.BindFlags =      D3D11_BIND_VERTEX_BUFFER;
    bd.CPUAccessFlags = 0;
    bd.MiscFlags =      0;
    hr = pd3dDevice->CreateBuffer( &bd, &InitData, &g_pFullscreenQuadVB );
    if( FAILED( hr ) )
    {
        OutputDebugString(L"Failed to create fullscreen quad vertex buffer.\n");
        return hr;
    }


    //
    // Vertex layouts
    //

    // Create simple vertex input layout
    const D3D11_INPUT_ELEMENT_DESC simplevertexlayout[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,  D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
    V_RETURN( pd3dDevice->CreateInputLayout( simplevertexlayout, ARRAYSIZE( simplevertexlayout ), 
                                             pBlobVSPassThrough->GetBufferPointer(), pBlobVSPassThrough->GetBufferSize(), 
                                             &g_pSimpleVertexLayout ) );
    
    // Create extended vertex input layout
    const D3D11_INPUT_ELEMENT_DESC extendedvertexlayout[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,  D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    0, 24, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
    V_RETURN( pd3dDevice->CreateInputLayout( extendedvertexlayout, ARRAYSIZE( extendedvertexlayout ), 
                                             pBlobVS->GetBufferPointer(), pBlobVS->GetBufferSize(), 
                                             &g_pExtendedVertexLayout ) );

    // Release blobs
    SAFE_RELEASE( pBlobVSPassThrough );
    SAFE_RELEASE( pBlobVS );
    SAFE_RELEASE( pBlobPS_LightingAndTexturing );

    //
    // Constant buffers
    //

    // Create main constant buffer
    bd.Usage = D3D11_USAGE_DYNAMIC;
    bd.ByteWidth = sizeof( GLOBAL_CONSTANT_BUFFER_STRUCT );
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    bd.MiscFlags = 0;
    hr = pd3dDevice->CreateBuffer( &bd, NULL, &g_pMainCB );
    if( FAILED( hr ) )
    {
        OutputDebugString(L"Failed to create constant buffer.\n");
        return hr;
    }

    // Create cube constant buffers
    bd.ByteWidth = sizeof( PER_MESH_CONSTANT_BUFFER_STRUCT );
    for (int i=0; i<MAX_NUMBER_OF_CUBES; i++)
    {
        pd3dDevice->CreateBuffer( &bd, NULL, &g_pCubeCB[i]);
    }

    // Create other mesh constant buffers
    pd3dDevice->CreateBuffer( &bd, NULL, &g_pOpaqueMeshCB);
    pd3dDevice->CreateBuffer( &bd, NULL, &g_pTransparentMeshCB);

    // Create tile coordinates constant buffer
    bd.ByteWidth = sizeof( TILE_COORDINATES_CONSTANT_BUFFER_STRUCT );
    pd3dDevice->CreateBuffer( &bd, NULL, &g_pTileCoordinatesCB);


    //
    // Load textures
    //

    // Diffuse map for cubes
    D3DX11_IMAGE_INFO ImageInfo;
    hr = D3DX11GetImageInfoFromFile( L"media\\cats_512x512.bmp", NULL, &ImageInfo, NULL);
      hr = D3DX11CreateShaderResourceViewFromFile( pd3dDevice, L"media\\cats_512x512.bmp", NULL, NULL, 
                                                   &g_pDiffuseTextureRV, NULL );
    if( FAILED( hr ) )
        return hr;

    // Default white texture for meshes with no textures
      hr = D3DX11CreateShaderResourceViewFromFile( pd3dDevice, L"media\\white.bmp", NULL, NULL, 
                                                   &g_pWhiteTextureRV, NULL );
    if( FAILED( hr ) )
        return hr;

    
    //
    // State objects
    //
    
    // Create solid and wireframe rasterizer state objects
    D3D11_RASTERIZER_DESC RasterDesc;
    ZeroMemory( &RasterDesc, sizeof(D3D11_RASTERIZER_DESC) );
    RasterDesc.FillMode = D3D11_FILL_SOLID;
    RasterDesc.CullMode = D3D11_CULL_NONE;
    RasterDesc.DepthClipEnable = TRUE;
    RasterDesc.MultisampleEnable = TRUE;
    RasterDesc.ScissorEnable = TRUE;    // New
    V_RETURN( pd3dDevice->CreateRasterizerState( &RasterDesc, &g_pRasterizerStateSolid ) );
    RasterDesc.FillMode = D3D11_FILL_WIREFRAME;
    V_RETURN( pd3dDevice->CreateRasterizerState( &RasterDesc, &g_pRasterizerStateWireframe ) );

    // Create sampler states
    D3D11_SAMPLER_DESC SSDesc;
    ZeroMemory( &SSDesc, sizeof( D3D11_SAMPLER_DESC ) );
    SSDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    SSDesc.AddressU         = D3D11_TEXTURE_ADDRESS_WRAP;
    SSDesc.AddressV         = D3D11_TEXTURE_ADDRESS_WRAP;
    SSDesc.AddressW         = D3D11_TEXTURE_ADDRESS_WRAP;
    SSDesc.ComparisonFunc   = D3D11_COMPARISON_NEVER;
    SSDesc.MaxAnisotropy    = 16;
    SSDesc.MinLOD           = 0;
    SSDesc.MaxLOD           = D3D11_FLOAT32_MAX;
    V_RETURN( pd3dDevice->CreateSamplerState( &SSDesc, &g_pSamplerStateLinearWrap) );
    SSDesc.AddressU         = D3D11_TEXTURE_ADDRESS_CLAMP;
    SSDesc.AddressV         = D3D11_TEXTURE_ADDRESS_CLAMP;
    SSDesc.AddressW         = D3D11_TEXTURE_ADDRESS_CLAMP;
    V_RETURN( pd3dDevice->CreateSamplerState( &SSDesc, &g_pSamplerStateLinearClamp) );
    SSDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    V_RETURN( pd3dDevice->CreateSamplerState( &SSDesc, &g_pSamplerStatePointClamp) );

    // Create a blend state to disable alpha blending
    D3D11_BLEND_DESC BlendState;
    ZeroMemory(&BlendState, sizeof(D3D11_BLEND_DESC));
    BlendState.IndependentBlendEnable      = FALSE;
    BlendState.RenderTarget[0].BlendEnable = FALSE;
    BlendState.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    hr = pd3dDevice->CreateBlendState(&BlendState, &g_pBlendStateNoBlend);

    // Create a blend state to enable alpha blending
    ZeroMemory(&BlendState, sizeof(D3D11_BLEND_DESC));
    BlendState.IndependentBlendEnable = TRUE;
    BlendState.RenderTarget[0].BlendEnable = TRUE;
    BlendState.RenderTarget[0].BlendOp               = D3D11_BLEND_OP_ADD;
    BlendState.RenderTarget[0].SrcBlend              = D3D11_BLEND_SRC_ALPHA;
    BlendState.RenderTarget[0].DestBlend             = D3D11_BLEND_INV_SRC_ALPHA;
    BlendState.RenderTarget[0].BlendOpAlpha          = D3D11_BLEND_OP_ADD;
    BlendState.RenderTarget[0].SrcBlendAlpha         = D3D11_BLEND_ONE;
    BlendState.RenderTarget[0].DestBlendAlpha        = D3D11_BLEND_ZERO;
    BlendState.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    hr = pd3dDevice->CreateBlendState(&BlendState, &g_pBlendStateSrcAlphaInvSrcAlphaBlend);

    // Create a blend state to enable background blending after under-blending
    /*
    ZeroMemory(&BlendState, sizeof(D3D11_BLEND_DESC));
    BlendState.IndependentBlendEnable = TRUE;
    BlendState.RenderTarget[0].BlendEnable = TRUE;
    BlendState.RenderTarget[0].BlendOp               = D3D11_BLEND_OP_ADD;
    BlendState.RenderTarget[0].SrcBlend              = D3D11_BLEND_ONE;
    BlendState.RenderTarget[0].DestBlend             = D3D11_BLEND_SRC_ALPHA;
    BlendState.RenderTarget[0].BlendOpAlpha          = D3D11_BLEND_OP_ADD;
    BlendState.RenderTarget[0].SrcBlendAlpha         = D3D11_BLEND_ONE;
    BlendState.RenderTarget[0].DestBlendAlpha        = D3D11_BLEND_ZERO;
    BlendState.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    hr = pd3dDevice->CreateBlendState(&BlendState, &g_pBlendStateBackgroundUnderBlending);
    */
    
    // Create a blend state to disable color writes
    BlendState.RenderTarget[0].SrcBlend  = D3D11_BLEND_ONE;
    BlendState.RenderTarget[0].DestBlend = D3D11_BLEND_ZERO;
    BlendState.RenderTarget[0].RenderTargetWriteMask = 0;
    hr = pd3dDevice->CreateBlendState(&BlendState, &g_pColorWritesOff);

    // Create depthstencil states
    D3D11_DEPTH_STENCIL_DESC DSDesc;
    DSDesc.DepthEnable                  = FALSE;
    DSDesc.DepthFunc                    = D3D11_COMPARISON_LESS_EQUAL;
    DSDesc.DepthWriteMask               = D3D11_DEPTH_WRITE_MASK_ZERO;
    DSDesc.StencilEnable                = FALSE;
    DSDesc.StencilReadMask              = 0xff;
    DSDesc.StencilWriteMask             = 0xff;
    DSDesc.FrontFace.StencilFailOp      = D3D11_STENCIL_OP_KEEP;
    DSDesc.FrontFace.StencilDepthFailOp = D3D11_STENCIL_OP_KEEP;
    DSDesc.FrontFace.StencilPassOp      = D3D11_STENCIL_OP_KEEP;
    DSDesc.FrontFace.StencilFunc        = D3D11_COMPARISON_ALWAYS;
    DSDesc.BackFace.StencilFailOp       = D3D11_STENCIL_OP_KEEP;
    DSDesc.BackFace.StencilDepthFailOp  = D3D11_STENCIL_OP_KEEP;
    DSDesc.BackFace.StencilPassOp       = D3D11_STENCIL_OP_KEEP;
    DSDesc.BackFace.StencilFunc         = D3D11_COMPARISON_ALWAYS;
    hr = pd3dDevice->CreateDepthStencilState(&DSDesc, &g_pDepthTestDisabledDSS);
    DSDesc.DepthEnable                  = TRUE;
    DSDesc.DepthFunc                    = D3D11_COMPARISON_LESS_EQUAL;
    DSDesc.DepthWriteMask               = D3D11_DEPTH_WRITE_MASK_ZERO;
    DSDesc.StencilEnable                = FALSE;
    hr = pd3dDevice->CreateDepthStencilState(&DSDesc, &g_pDepthTestEnabledNoDepthWritesDSS);
    DSDesc.DepthFunc                    = D3D11_COMPARISON_LESS_EQUAL;
    DSDesc.DepthWriteMask               = D3D11_DEPTH_WRITE_MASK_ALL;
    hr = pd3dDevice->CreateDepthStencilState(&DSDesc, &g_pDepthTestEnabledDSS);
    DSDesc.DepthEnable                  = TRUE;
    DSDesc.DepthFunc                    = D3D11_COMPARISON_LESS_EQUAL;
    DSDesc.DepthWriteMask               = D3D11_DEPTH_WRITE_MASK_ZERO;
    DSDesc.StencilEnable                = TRUE;
    DSDesc.StencilReadMask              = 0xFF;
    DSDesc.StencilWriteMask             = 0xFF;
    DSDesc.FrontFace.StencilFailOp      = D3D11_STENCIL_OP_KEEP;
    DSDesc.FrontFace.StencilDepthFailOp = D3D11_STENCIL_OP_KEEP;
    DSDesc.FrontFace.StencilPassOp      = D3D11_STENCIL_OP_INCR_SAT;
    DSDesc.FrontFace.StencilFunc        = D3D11_COMPARISON_ALWAYS;
    DSDesc.BackFace.StencilFailOp       = D3D11_STENCIL_OP_KEEP;
    DSDesc.BackFace.StencilDepthFailOp  = D3D11_STENCIL_OP_KEEP;
    DSDesc.BackFace.StencilPassOp       = D3D11_STENCIL_OP_INCR_SAT;
    DSDesc.BackFace.StencilFunc         = D3D11_COMPARISON_ALWAYS;
    hr = pd3dDevice->CreateDepthStencilState(&DSDesc, &g_pDepthTestEnabledNoDepthWritesStencilWriteIncrementDSS);
    DSDesc.DepthEnable                  = FALSE;
    DSDesc.DepthFunc                    = D3D11_COMPARISON_LESS_EQUAL;
    DSDesc.DepthWriteMask               = D3D11_DEPTH_WRITE_MASK_ZERO;
    DSDesc.StencilEnable                = TRUE;
    DSDesc.StencilReadMask              = 0xFF;
    DSDesc.StencilWriteMask             = 0x00;
    DSDesc.FrontFace.StencilFailOp      = D3D11_STENCIL_OP_KEEP;
    DSDesc.FrontFace.StencilDepthFailOp = D3D11_STENCIL_OP_KEEP;
    DSDesc.FrontFace.StencilPassOp      = D3D11_STENCIL_OP_KEEP;
    DSDesc.FrontFace.StencilFunc        = D3D11_COMPARISON_LESS;
    DSDesc.BackFace.StencilFailOp       = D3D11_STENCIL_OP_KEEP;
    DSDesc.BackFace.StencilDepthFailOp  = D3D11_STENCIL_OP_KEEP;
    DSDesc.BackFace.StencilPassOp       = D3D11_STENCIL_OP_KEEP;
    DSDesc.BackFace.StencilFunc         = D3D11_COMPARISON_LESS;
    hr = pd3dDevice->CreateDepthStencilState(&DSDesc, &g_pDepthTestDisabledStencilTestLessDSS);

    // Create event query for flushing the pipeline
    D3D11_QUERY_DESC QueryDesc = { D3D11_QUERY_EVENT, 0 };
    pd3dDevice->CreateQuery(&QueryDesc, &g_pEventQuery);

    return S_OK;
}


//--------------------------------------------------------------------------------------
// Create any D3D11 resources that depend on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D11ResizedSwapChain( ID3D11Device* pd3dDevice, IDXGISwapChain* pSwapChain,
                                          const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext )
{
    HRESULT hr;

    V_RETURN( g_DialogResourceManager.OnD3D11ResizedSwapChain( pd3dDevice, pBackBufferSurfaceDesc ) );
    V_RETURN( g_D3DSettingsDlg.OnD3D11ResizedSwapChain( pd3dDevice, pBackBufferSurfaceDesc ) );

    // Copy back buffer description into global one
    g_pBackBufferSurfaceDesc = *pBackBufferSurfaceDesc;

    // Set GUI size and locations
    g_HUD.SetLocation( pBackBufferSurfaceDesc->Width - 170, 0 );
    g_HUD.SetSize( 170, 170 );
    g_SampleUI.SetLocation( pBackBufferSurfaceDesc->Width - 180, pBackBufferSurfaceDesc->Height - 330 );
    g_SampleUI.SetSize( 180, 150 );

    // Create resources
    CreateResources(pd3dDevice, DXUTGetD3D11DeviceContext(), pBackBufferSurfaceDesc);

    // Setup the camera's projection parameters
    float fAspectRatio = pBackBufferSurfaceDesc->Width / (FLOAT)pBackBufferSurfaceDesc->Height;
    g_Camera.SetProjParams(  D3DX_PI / 4, fAspectRatio, 0.1f,  50.0f );
    g_LCamera.SetProjParams( D3DX_PI / 4, fAspectRatio, 10.0f, 500.0f );
    

    // Set default viewport
    D3D11_VIEWPORT Viewport;
    Viewport.Width      = (float)pBackBufferSurfaceDesc->Width;
    Viewport.Height     = (float)pBackBufferSurfaceDesc->Height;
    Viewport.TopLeftX   = 0.0f;
    Viewport.TopLeftY   = 0.0f;
    Viewport.MinDepth   = 0.0f;
    Viewport.MaxDepth   = 1.0f;
    DXUTGetD3D11DeviceContext()->RSSetViewports(1, &Viewport);

    // Set default scissor rect
    D3D11_RECT ScissorRect;
    ScissorRect.top  = 0;
    ScissorRect.left = 0;
    ScissorRect.bottom = pBackBufferSurfaceDesc->Height;
    ScissorRect.right  = pBackBufferSurfaceDesc->Width;
    DXUTGetD3D11DeviceContext()->RSSetScissorRects(1, &ScissorRect);

    // Save back buffer size
    g_fScreenWidth  = float(pBackBufferSurfaceDesc->Width);
    g_fScreenHeight = float(pBackBufferSurfaceDesc->Height);

    return S_OK;
}


//--------------------------------------------------------------------------------------
// Create OIT-related resources
//--------------------------------------------------------------------------------------
HRESULT CreateResources(ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext, 
                        const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc)
{
    D3D11_TEXTURE2D_DESC            TexDesc;
    D3D11_SHADER_RESOURCE_VIEW_DESC SRVBufferDesc;
    D3D11_RENDER_TARGET_VIEW_DESC   RTVDesc;
    const DXGI_SAMPLE_DESC          SingleSample = { 1, 0 };
    HRESULT                         hr;

    // Event query to ensure all resources have finished being used used before being destroyed
    pd3dImmediateContext->End(g_pEventQuery);
    while( pd3dImmediateContext->GetData(g_pEventQuery, NULL, 0, 0) != S_OK );

    // Destroy existing resources
    DestroyResources();

    //
    // Create resources common to all methods
    //

    // Create main render target (with MSAA or not depending on whether this was selected)
    TexDesc.Width =          pBackBufferSurfaceDesc->Width;
    TexDesc.Height =         pBackBufferSurfaceDesc->Height;
    TexDesc.Format =         DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    TexDesc.SampleDesc =     g_MSAASampleDesc;
    TexDesc.MipLevels =      1;
    TexDesc.Usage =          D3D11_USAGE_DEFAULT;
    TexDesc.MiscFlags =      0;
    TexDesc.CPUAccessFlags = 0;
    TexDesc.BindFlags =      D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    TexDesc.ArraySize =      1;
    hr = pd3dDevice->CreateTexture2D(&TexDesc, NULL, &g_pMainRenderTargetTexture);

    // Create SRV for g_pMainRenderTargetTexture
    SRVBufferDesc.Format         = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    SRVBufferDesc.ViewDimension  = 
        g_MSAASampleDesc.Count>1 ? D3D11_SRV_DIMENSION_TEXTURE2DMS : D3D11_SRV_DIMENSION_TEXTURE2D;
    SRVBufferDesc.Texture2D.MipLevels       = 1;
    SRVBufferDesc.Texture2D.MostDetailedMip = 0;
    hr = pd3dDevice->CreateShaderResourceView(g_pMainRenderTargetTexture, &SRVBufferDesc, 
                                              &g_pMainRenderTargetTextureSRV);

    // Create RTV for g_pMainRenderTargetTexture
    RTVDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    RTVDesc.ViewDimension = g_MSAASampleDesc.Count>1 ? D3D11_RTV_DIMENSION_TEXTURE2DMS : D3D11_RTV_DIMENSION_TEXTURE2D;
    RTVDesc.Texture2D.MipSlice = 0;
    hr = pd3dDevice->CreateRenderTargetView(g_pMainRenderTargetTexture, &RTVDesc, &g_pMainRenderTargetTextureRTV);

    // Also create a copy of main render target texture
    hr = pd3dDevice->CreateTexture2D(&TexDesc, NULL, &g_pCopyOfMainRenderTargetTexture);
    hr = pd3dDevice->CreateShaderResourceView(g_pCopyOfMainRenderTargetTexture, &SRVBufferDesc, 
                                              &g_pCopyOfMainRenderTargetTextureSRV);
    hr = pd3dDevice->CreateRenderTargetView(g_pCopyOfMainRenderTargetTexture, &RTVDesc, 
                                            &g_pCopyOfMainRenderTargetTextureRTV);

    // Create non-MSAA version of resolved render target and its SRV
    TexDesc.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS;
    TexDesc.SampleDesc = SingleSample;
    TexDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET | D3D11_BIND_UNORDERED_ACCESS;
    hr = pd3dDevice->CreateTexture2D(&TexDesc, NULL, &g_pMainRenderTargetTextureResolved);
    SRVBufferDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    hr = pd3dDevice->CreateShaderResourceView(g_pMainRenderTargetTextureResolved, &SRVBufferDesc, 
                                              &g_pMainRenderTargetTextureResolvedSRV);

    // Create RTV for resolved render target
    RTVDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    RTVDesc.ViewDimension = g_MSAASampleDesc.Count>1 ? D3D11_RTV_DIMENSION_TEXTURE2DMS : D3D11_RTV_DIMENSION_TEXTURE2D;
    RTVDesc.Texture2D.MipSlice = 0;
    hr = pd3dDevice->CreateRenderTargetView(g_pMainRenderTargetTextureResolved, &RTVDesc, &g_pMainRenderTargetTextureResolvedRTV);

    // Also create a UAV for it (used with Compute Shader version)
    D3D11_UNORDERED_ACCESS_VIEW_DESC UAVBufferDesc;
    UAVBufferDesc.Format             = DXGI_FORMAT_R8G8B8A8_UNORM;
    UAVBufferDesc.ViewDimension      = D3D11_UAV_DIMENSION_TEXTURE2D;
    UAVBufferDesc.Texture2D.MipSlice = 0;
    hr = pd3dDevice->CreateUnorderedAccessView(g_pMainRenderTargetTextureResolved, &UAVBufferDesc, 
                                               &g_pMainRenderTargetTextureResolvedUAV);


    // Create a screen-sized STAGING resource of copy of main RT
    TexDesc.Usage          = D3D11_USAGE_STAGING;
    TexDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    TexDesc.SampleDesc     = SingleSample;
    TexDesc.BindFlags      = 0;
    hr = pd3dDevice->CreateTexture2D(&TexDesc, NULL, &g_pCopyOfMainRenderTargetTextureSTAGING);


    // Create a screen-sized depth stencil resource
    // Use a full 32-bits format for depth when depth peeling is used
    // This is to avoid Z-fighting artefacts due to the "manual" depth buffer implementation
    TexDesc.Width =          pBackBufferSurfaceDesc->Width;
    TexDesc.Height =         pBackBufferSurfaceDesc->Height;
    TexDesc.Format =         DXGI_FORMAT_R24G8_TYPELESS;
    TexDesc.SampleDesc =     g_MSAASampleDesc;
    TexDesc.MipLevels =      1;
    TexDesc.Usage =          D3D11_USAGE_DEFAULT;
    TexDesc.MiscFlags =      0;
    TexDesc.CPUAccessFlags = 0;
    TexDesc.BindFlags =      D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_DEPTH_STENCIL;
    TexDesc.ArraySize =      1;
    hr = pd3dDevice->CreateTexture2D(&TexDesc, NULL, &g_pDepthStencilTexture);

    // Create Depth Stencil View
    D3D11_DEPTH_STENCIL_VIEW_DESC SRVDepthStencilDesc;
    SRVDepthStencilDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    SRVDepthStencilDesc.ViewDimension = 
        g_MSAASampleDesc.Count>1 ? D3D11_DSV_DIMENSION_TEXTURE2DMS : D3D11_DSV_DIMENSION_TEXTURE2D;
    SRVDepthStencilDesc.Texture2D.MipSlice = 0;
    SRVDepthStencilDesc.Flags              = 0;
    hr = pd3dDevice->CreateDepthStencilView(g_pDepthStencilTexture, &SRVDepthStencilDesc, 
                                            &g_pDepthStencilTextureDSV);

    //
    // Resources for LINKED LIST method
    //
    // Creation of this very large resource goes first to avoid memory 
    // fragmentation issues that could push this surface onto nonlocal
    if (g_nOITMethod==OITMETHOD_LINKEDLIST)
    {
        UINT uTileWidth  = pBackBufferSurfaceDesc->Width;
        UINT uTileHeight = pBackBufferSurfaceDesc->Height;

        // Tiling mode only supported for linked list method
        if (g_nOITMethod==OITMETHOD_LINKEDLIST)
        {
            uTileWidth  = ( pBackBufferSurfaceDesc->Width  + (NUMBER_OF_TILES_X-1) ) / NUMBER_OF_TILES_X;
            uTileHeight = ( pBackBufferSurfaceDesc->Height + (NUMBER_OF_TILES_Y-1) ) / NUMBER_OF_TILES_Y;
        }

        // Create Fragment and Link Buffer as structured buffer
        // Use an average translucent overdraw thus the buffer must contain AVERAGE_TRANSLUCENT_OVERDRAW*width*height entries
        D3D11_BUFFER_DESC BufferDesc;
        BufferDesc.CPUAccessFlags          = 0;
        BufferDesc.BindFlags               = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        BufferDesc.ByteWidth               = (DWORD)(AVERAGE_TRANSLUCENT_OVERDRAW * uTileWidth * uTileHeight * 
                                                     sizeof(FRAGMENT_AND_LINK_BUFFER_STRUCT) );
        BufferDesc.MiscFlags               = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        BufferDesc.Usage                   = D3D11_USAGE_DEFAULT;
        BufferDesc.StructureByteStride     = sizeof(FRAGMENT_AND_LINK_BUFFER_STRUCT);
        V_RETURN( pd3dDevice->CreateBuffer(&BufferDesc, NULL, &g_pFragmentAndLinkStructuredBuffer) );

        // Create UAV view of Fragment and Link Buffer
        D3D11_UNORDERED_ACCESS_VIEW_DESC UAVDesc;
        UAVDesc.Format                = DXGI_FORMAT_UNKNOWN;
        UAVDesc.ViewDimension         = D3D11_UAV_DIMENSION_BUFFER;
        UAVDesc.Buffer.FirstElement   = 0;
        UAVDesc.Buffer.NumElements    = (DWORD)(AVERAGE_TRANSLUCENT_OVERDRAW * uTileWidth * uTileHeight);
        UAVDesc.Buffer.Flags          = D3D11_BUFFER_UAV_FLAG_COUNTER;
        hr = pd3dDevice->CreateUnorderedAccessView(g_pFragmentAndLinkStructuredBuffer, &UAVDesc, 
                                                   &g_pFragmentAndLinkStructuredBufferUAV);

        // Create SRV view of Fragment and Link Buffer
        SRVBufferDesc.Format                = DXGI_FORMAT_UNKNOWN;
        SRVBufferDesc.ViewDimension         = D3D11_SRV_DIMENSION_BUFFER;
        SRVBufferDesc.Buffer.ElementOffset  = 0;
        SRVBufferDesc.Buffer.ElementWidth   = (DWORD)(AVERAGE_TRANSLUCENT_OVERDRAW * uTileWidth * uTileHeight);
        hr = pd3dDevice->CreateShaderResourceView(g_pFragmentAndLinkStructuredBuffer, &SRVBufferDesc, 
                                                  &g_pFragmentAndLinkStructuredBufferSRV);
    }

    // Start Offset buffer is used in linked list, Precalc and fixed overdraw method
    if (g_nOITMethod==OITMETHOD_LINKEDLIST)
    {
        UINT uTileWidth  = pBackBufferSurfaceDesc->Width;
        UINT uTileHeight = pBackBufferSurfaceDesc->Height;

        // Tiling mode only supported for linked list method
        if (g_nOITMethod==OITMETHOD_LINKEDLIST)
        {
            uTileWidth  = ( pBackBufferSurfaceDesc->Width  + (NUMBER_OF_TILES_X-1) ) / NUMBER_OF_TILES_X;
            uTileHeight = ( pBackBufferSurfaceDesc->Height + (NUMBER_OF_TILES_Y-1) ) / NUMBER_OF_TILES_Y;
        }

        // Create Start Offset buffer
        D3D11_BUFFER_DESC OffsetBufferDesc;
        OffsetBufferDesc.BindFlags           = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        OffsetBufferDesc.ByteWidth           = uTileWidth * uTileHeight * sizeof(UINT);
        OffsetBufferDesc.MiscFlags           = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        OffsetBufferDesc.Usage               = D3D11_USAGE_DEFAULT;
        OffsetBufferDesc.CPUAccessFlags      = 0;
        OffsetBufferDesc.StructureByteStride = 0;
        V_RETURN( pd3dDevice->CreateBuffer(&OffsetBufferDesc, NULL, &g_pStartOffsetBuffer) );

        // Create UAV view of Start Offset buffer
        D3D11_UNORDERED_ACCESS_VIEW_DESC UAVOffsetBufferDesc;
        UAVOffsetBufferDesc.Format              = DXGI_FORMAT_R32_TYPELESS;
        UAVOffsetBufferDesc.ViewDimension       = D3D11_UAV_DIMENSION_BUFFER;
        UAVOffsetBufferDesc.Buffer.FirstElement = 0;
        UAVOffsetBufferDesc.Buffer.NumElements  = uTileWidth * uTileHeight;
        UAVOffsetBufferDesc.Buffer.Flags        = D3D11_BUFFER_UAV_FLAG_RAW;
        hr = pd3dDevice->CreateUnorderedAccessView(g_pStartOffsetBuffer, &UAVOffsetBufferDesc, &g_pStartOffsetBufferUAV);

        // Create SRV view of Start Offset buffer
        D3D11_SHADER_RESOURCE_VIEW_DESC SRVOffsetBufferDesc;
        SRVOffsetBufferDesc.Format              = DXGI_FORMAT_R32_UINT;
        SRVOffsetBufferDesc.ViewDimension       = D3D11_SRV_DIMENSION_BUFFER;
        SRVOffsetBufferDesc.Buffer.ElementOffset= 0;
        SRVOffsetBufferDesc.Buffer.ElementWidth = (DWORD)(uTileWidth * uTileHeight);
        hr = pd3dDevice->CreateShaderResourceView(g_pStartOffsetBuffer, &SRVOffsetBufferDesc, &g_pStartOffsetBufferSRV);
    }

    //
    // Create OIT-specific shaders
    //
    ID3DBlob* pErrorBlob = NULL;
    DWORD     dwShaderFlags = D3D10_SHADER_ENABLE_STRICTNESS;
#if defined( DEBUG ) || defined( _DEBUG )
    dwShaderFlags |= D3D10_SHADER_DEBUG;
#endif

    // Prepare macro string 
    char pszValue[4];
    _itoa_s(g_MSAASampleDesc.Count, pszValue, sizeof(pszValue), 10);
    D3D10_SHADER_MACRO    pEffectMacrosPS[] = { "NUM_SAMPLES", pszValue, NULL, NULL };

    // Pixel shaders for linked list method
    if (g_nOITMethod==OITMETHOD_LINKEDLIST)
    {
        ID3DBlob* pBlobPS_StoreFragments = NULL;
        hr = D3DX11CompileFromFile( L"PS_OIT_LinkedList.hlsl", pEffectMacrosPS, NULL, "PS_StoreFragments", "ps_5_0", 
                                    dwShaderFlags, 0, NULL, &pBlobPS_StoreFragments, &pErrorBlob, NULL);
        if( FAILED(hr) )
        {
            OutputDebugStringA( (char*)pErrorBlob->GetBufferPointer() );
            SAFE_RELEASE( pErrorBlob );
            return hr;
        }
        V_RETURN( pd3dDevice->CreatePixelShader( pBlobPS_StoreFragments->GetBufferPointer(), pBlobPS_StoreFragments->GetBufferSize(),  
                                                 NULL, &g_pStoreFragmentsPS_LinkedList ) );
        SAFE_RELEASE( pBlobPS_StoreFragments );
        
        ID3DBlob* pBlobPS_RenderFragments = NULL;
        hr = D3DX11CompileFromFile( L"PS_OIT_LinkedList.hlsl", pEffectMacrosPS, NULL, "PS_RenderFragments", "ps_5_0", 
                                    dwShaderFlags, 0, NULL, &pBlobPS_RenderFragments, &pErrorBlob, NULL);
        if( FAILED(hr) )
        {
            OutputDebugStringA( (char*)pErrorBlob->GetBufferPointer() );
            SAFE_RELEASE( pErrorBlob );
            return hr;
        }
        V_RETURN( pd3dDevice->CreatePixelShader( pBlobPS_RenderFragments->GetBufferPointer(), pBlobPS_RenderFragments->GetBufferSize(), 
                                                 NULL, &g_pRenderFragmentsPS_LinkedList ) );
        SAFE_RELEASE( pBlobPS_RenderFragments );
        
        if (g_MSAASampleDesc.Count>1)
        {
            ID3DBlob* pBlobPS_RenderFragmentsWithResolve = NULL;
            hr = D3DX11CompileFromFile( L"PS_OIT_LinkedList.hlsl", pEffectMacrosPS, NULL, "PS_RenderFragmentsWithResolve", "ps_5_0", 
                                        dwShaderFlags, 0, NULL, &pBlobPS_RenderFragmentsWithResolve, &pErrorBlob, NULL);
            if( FAILED( hr ) )
            {
                OutputDebugStringA( (char*)pErrorBlob->GetBufferPointer() );
                SAFE_RELEASE( pErrorBlob );
                return hr;
            }
            V_RETURN( pd3dDevice->CreatePixelShader( pBlobPS_RenderFragmentsWithResolve->GetBufferPointer(), 
                                                     pBlobPS_RenderFragmentsWithResolve->GetBufferSize(), 
                                                     NULL, &g_pRenderFragmentsWithResolvePS_LinkedList ) );
            SAFE_RELEASE( pBlobPS_RenderFragmentsWithResolve );
        }
    }

    return D3D_OK;
}


//--------------------------------------------------------------------------------------
// Destroy OIT-related resources
//--------------------------------------------------------------------------------------
void DestroyResources()
{
    // Shaders
    SAFE_RELEASE(g_pStoreFragmentsPS_LinkedList);
    SAFE_RELEASE(g_pRenderFragmentsPS_LinkedList);
    SAFE_RELEASE(g_pRenderFragmentsWithResolvePS_LinkedList);

    // Main render target
    SAFE_RELEASE(g_pMainRenderTargetTextureRTV);
    SAFE_RELEASE(g_pMainRenderTargetTextureSRV);
    SAFE_RELEASE(g_pMainRenderTargetTexture);
    
    // Copy of main render target
    SAFE_RELEASE(g_pCopyOfMainRenderTargetTextureRTV);
    SAFE_RELEASE(g_pCopyOfMainRenderTargetTextureSRV);
    SAFE_RELEASE(g_pCopyOfMainRenderTargetTexture);
    SAFE_RELEASE(g_pCopyOfMainRenderTargetTextureSTAGING);
    
    // Resolved main render target
    SAFE_RELEASE(g_pMainRenderTargetTextureResolvedUAV);
    SAFE_RELEASE(g_pMainRenderTargetTextureResolvedRTV);
    SAFE_RELEASE(g_pMainRenderTargetTextureResolvedSRV);
    SAFE_RELEASE(g_pMainRenderTargetTextureResolved);

    // Depth stencil texture
    SAFE_RELEASE(g_pDepthStencilTextureDSV);
    SAFE_RELEASE(g_pDepthStencilTexture);
    
    // Start offset buffer
    SAFE_RELEASE(g_pStartOffsetBufferSRV);
    SAFE_RELEASE(g_pStartOffsetBufferUAV);
    SAFE_RELEASE(g_pStartOffsetBuffer);

    // Fragment and link buffer (linked list method)
    SAFE_RELEASE(g_pFragmentAndLinkStructuredBufferUAV);
    SAFE_RELEASE(g_pFragmentAndLinkStructuredBufferSRV);
    SAFE_RELEASE(g_pFragmentAndLinkStructuredBuffer);
}

//--------------------------------------------------------------------------------------
// Render the scene using the D3D11 device
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11FrameRender( ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext, 
                                  double fTime, float fElapsedTime, void* pUserContext )
{
    HRESULT                     hr;
    UINT                        stride;
    UINT                        offset = 0;
    ID3D11RenderTargetView*     pRTV[1] = { NULL };
    UINT                        pUAVCounters[4] = { 0, 0, 0, 0 };
    ID3D11UnorderedAccessView*  pUAV[4] = { NULL, NULL, NULL, NULL };
    static DWORD                dwFrameNumber = 1;
    
    // If the settings dialog is being shown, then render it instead of rendering the app's scene
    if( g_D3DSettingsDlg.IsActive() )
    {
        g_D3DSettingsDlg.OnRender( fElapsedTime );
        return;
    }

    // Recreate method-specific resources if required
    if (g_bRecreateResources)
    {
        CreateResources(pd3dDevice, pd3dImmediateContext, &g_pBackBufferSurfaceDesc);
        g_bRecreateResources = FALSE;
    }

    // Set render target to main render target
    pRTV[0] = g_pMainRenderTargetTextureRTV;
    pd3dImmediateContext->OMSetRenderTargets(1, pRTV, g_pDepthStencilTextureDSV );

    // Clear the render target and depth stencil
    float ClearColor[4] = { 0.05f, 0.05f, 0.05f, 1.0f };
    pd3dImmediateContext->ClearRenderTargetView( g_pMainRenderTargetTextureRTV, ClearColor );
    pd3dImmediateContext->ClearDepthStencilView( g_pDepthStencilTextureDSV, 
                                                 D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0, 0 );

    // Update main constant buffer
    D3DXMATRIX mTransposedView;
    D3DXMatrixTranspose(&mTransposedView, g_Camera.GetViewMatrix());
    D3DXMATRIX mTransposedProj;
    D3DXMatrixTranspose(&mTransposedProj, g_Camera.GetProjMatrix());
    D3DXMATRIX mViewProjection = (*g_Camera.GetViewMatrix()) * (*g_Camera.GetProjMatrix());    
    D3DXMATRIX mTransposedViewProjection;
    D3DXMatrixTranspose(&mTransposedViewProjection, &mViewProjection);    
    D3D11_MAPPED_SUBRESOURCE MappedSubResource;
    pd3dImmediateContext->Map(g_pMainCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedSubResource );
    ((GLOBAL_CONSTANT_BUFFER_STRUCT *)MappedSubResource.pData)->mView =           mTransposedView;
    ((GLOBAL_CONSTANT_BUFFER_STRUCT *)MappedSubResource.pData)->mProjection =     mTransposedProj;
    ((GLOBAL_CONSTANT_BUFFER_STRUCT *)MappedSubResource.pData)->mViewProjection = mTransposedViewProjection;
    D3DXVECTOR4 vLight(-0.577f, 2.577f, -0.577f, 1.0f);
    D3DXVec4Transform( &vLight, &vLight, g_LCamera.GetWorldMatrix() );
    D3DXVec4Normalize(&vLight, &vLight);
    ((GLOBAL_CONSTANT_BUFFER_STRUCT *)MappedSubResource.pData)->vLightVector = vLight;
    D3DXVECTOR3 vViewVector = *g_Camera.GetLookAtPt() - *g_Camera.GetEyePt();
    D3DXVec3Normalize(&vViewVector, &vViewVector);
    ((GLOBAL_CONSTANT_BUFFER_STRUCT *)MappedSubResource.pData)->vViewVector = D3DXVECTOR4(vViewVector, 1.0f);
    ((GLOBAL_CONSTANT_BUFFER_STRUCT *)MappedSubResource.pData)->vScreenDimensions =    
        D3DXVECTOR4(g_fScreenWidth, g_fScreenHeight, 1.0f/g_fScreenWidth, 1.0f/g_fScreenHeight);
    pd3dImmediateContext->Unmap(g_pMainCB, 0);

    // Bind the main constant buffer at slot 0 for all stages
    ID3D11Buffer* pBuffers[1] = { g_pMainCB };
    pd3dImmediateContext->VSSetConstantBuffers( 0, 1, pBuffers );
    pd3dImmediateContext->PSSetConstantBuffers( 0, 1, pBuffers );
    pd3dImmediateContext->CSSetConstantBuffers( 0, 1, pBuffers );

    // Update room constant buffer
    D3DXMATRIX mWorld;
    D3DXMatrixIdentity(&mWorld);
    D3DXMATRIX mTransposedWorld;
    D3DXMatrixTranspose(&mTransposedWorld, &mWorld);
    D3DXMATRIX mWorldViewProjection;
    mWorldViewProjection = mWorld * mViewProjection;
    D3DXMATRIX mTransposedWorldViewProjection;
    D3DXMatrixTranspose(&mTransposedWorldViewProjection, &mWorldViewProjection);
    pd3dImmediateContext->Map(g_pOpaqueMeshCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedSubResource );
    ((PER_MESH_CONSTANT_BUFFER_STRUCT *)MappedSubResource.pData)->mWorld                = mTransposedWorld;
    ((PER_MESH_CONSTANT_BUFFER_STRUCT *)MappedSubResource.pData)->mWorldViewProjection  = mTransposedWorldViewProjection;
    ((PER_MESH_CONSTANT_BUFFER_STRUCT *)MappedSubResource.pData)->vMeshColor = D3DXVECTOR4(1.0, 1.0f, 1.0f, 1.0f);
    pd3dImmediateContext->Unmap(g_pOpaqueMeshCB, 0);

    // Update transparent model constant buffer
    D3DXMatrixIdentity(&mWorld);
    D3DXMATRIX mScale;
    float fUniformScaleFactor = vTransparentMeshesInScene[g_nCurrentModel].fScaleFactor;
    D3DXMatrixScaling(&mScale, fUniformScaleFactor, fUniformScaleFactor, fUniformScaleFactor);
    mWorld *= mScale;
    D3DXMATRIX mRotation;
    D3DXMatrixRotationAxis(&mRotation, (const D3DXVECTOR3 *)&vTransparentMeshesInScene[g_nCurrentModel].vRotationAxisAndAngle, 
                                       vTransparentMeshesInScene[g_nCurrentModel].vRotationAxisAndAngle.w);
    mWorld *= mRotation;
    D3DXMATRIX mTranslation;
    D3DXMatrixTranslation(&mTranslation, vTransparentMeshesInScene[g_nCurrentModel].vWorldSpacePosition.x, 
                                         vTransparentMeshesInScene[g_nCurrentModel].vWorldSpacePosition.y, 
                                         vTransparentMeshesInScene[g_nCurrentModel].vWorldSpacePosition.z);
    mWorld *= mTranslation;
    D3DXMatrixTranspose(&mTransposedWorld, &mWorld);
    mWorldViewProjection = mWorld * mViewProjection;
    D3DXMatrixTranspose(&mTransposedWorldViewProjection, &mWorldViewProjection);
    pd3dImmediateContext->Map(g_pTransparentMeshCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedSubResource );
    ((PER_MESH_CONSTANT_BUFFER_STRUCT *)MappedSubResource.pData)->mWorld                = mTransposedWorld;
    ((PER_MESH_CONSTANT_BUFFER_STRUCT *)MappedSubResource.pData)->mWorldViewProjection  = mTransposedWorldViewProjection;
    ((PER_MESH_CONSTANT_BUFFER_STRUCT *)MappedSubResource.pData)->vMeshColor = vTransparentMeshesInScene[g_nCurrentModel].vMeshColor; 
    pd3dImmediateContext->Unmap(g_pTransparentMeshCB, 0);

    // Solid or wireframe rendering
    pd3dImmediateContext->RSSetState( g_bEnableWireFrame ? g_pRasterizerStateWireframe : g_pRasterizerStateSolid);


    //
    // Render room
    // 

    // Set shaders
    pd3dImmediateContext->VSSetShader( g_pMainVS, NULL, 0 );
    pd3dImmediateContext->PSSetShader( g_pLightingAndTexturingPS, NULL, 0 ); 

    // Disable blending
    pd3dImmediateContext->OMSetBlendState(g_pBlendStateNoBlend, 0, 0xffffffff);

    // Set Depth Stencil state
    pd3dImmediateContext->OMSetDepthStencilState(g_pDepthTestEnabledDSS, 0);

    // Bind the mesh constant buffer at slot 1 for all stages
    pBuffers[0] = g_pOpaqueMeshCB;
    pd3dImmediateContext->VSSetConstantBuffers( 1, 1, pBuffers );
    pd3dImmediateContext->PSSetConstantBuffers( 1, 1, pBuffers );

    // Set sampler state
    ID3D11SamplerState* pSS[1] = { g_pSamplerStateLinearWrap };
    pd3dImmediateContext->PSSetSamplers(0, 1, pSS);

    // Render room model
    RenderModel(pd3dImmediateContext, &g_OpaqueMesh);


    //
    // Render cubes
    //

    // Update constant buffers for all cubes
    UpdateCubeConstantBuffers(pd3dImmediateContext);
        
    // Set RT to back buffer
    pRTV[0] = g_pMainRenderTargetTextureRTV;
    pd3dImmediateContext->OMSetRenderTargets(1, pRTV, g_pDepthStencilTextureDSV );

    // Set shaders
    pd3dImmediateContext->VSSetShader( g_pMainVS, NULL, 0 );
    pd3dImmediateContext->PSSetShader( g_pLightingAndTexturingPS, NULL, 0 ); 

    // Set shader resources
    ID3D11ShaderResourceView* pSRV[4] = { g_pDiffuseTextureRV, NULL, NULL, NULL };
    pd3dImmediateContext->PSSetShaderResources(0, 4, pSRV);

    // Disable blending
    pd3dImmediateContext->OMSetBlendState(g_pBlendStateNoBlend, 0, 0xffffffff);

    // Set Depth Stencil state
    pd3dImmediateContext->OMSetDepthStencilState(g_pDepthTestEnabledDSS, 0);

    // Set sampler state
    pSS[0] = g_pSamplerStateLinearClamp;
    pd3dImmediateContext->PSSetSamplers(0, 1, pSS);

    // Render cubes geometry
    RenderCubes(pd3dImmediateContext);

    
    //
    // Copy current RT to texture (will be needed for resolve pass)
    //
    if (g_nOITMethod != OITMETHOD_NOSORTING)
    {
        // Copy current contents of render target onto texture
        pd3dImmediateContext->CopyResource(g_pCopyOfMainRenderTargetTexture, g_pMainRenderTargetTexture);
    }
    

    // No sorting method
    if (g_nOITMethod==OITMETHOD_NOSORTING)
    {
        // Set RT to back buffer
        pRTV[0] = g_pMainRenderTargetTextureRTV;
        pd3dImmediateContext->OMSetRenderTargets(1, pRTV, g_pDepthStencilTextureDSV );

        // Set shaders
        pd3dImmediateContext->VSSetShader( g_pMainVS, NULL, 0 );
        pd3dImmediateContext->PSSetShader( g_pLightingAndTexturingPS, NULL, 0 ); 

        // Set depth stencil state
        pd3dImmediateContext->OMSetDepthStencilState(g_pDepthTestEnabledNoDepthWritesDSS, 0x00);

        // Set blend mode to SRCALPHA-INVSRCALPHA
        pd3dImmediateContext->OMSetBlendState(g_pBlendStateSrcAlphaInvSrcAlphaBlend, 0, 0xffffffff);
        
        // Bind the mesh constant buffer at slot 1 for all stages
        pBuffers[0] = g_pTransparentMeshCB;
        pd3dImmediateContext->VSSetConstantBuffers( 1, 1, pBuffers );
        pd3dImmediateContext->PSSetConstantBuffers( 1, 1, pBuffers );

        // Render transparent model
        RenderModel(pd3dImmediateContext, &g_TransparentMesh[g_nCurrentModel]);

        // Resolve/copy onto back buffer
        ID3D11Resource* pBackBufferResource;
        DXUTGetD3D11RenderTargetView()->GetResource(&pBackBufferResource);
        if (g_MSAASampleDesc.Count>1)
        {
            pd3dImmediateContext->ResolveSubresource(pBackBufferResource, 0, g_pMainRenderTargetTexture, 0, 
                                                     DXGI_FORMAT_R8G8B8A8_UNORM_SRGB);
        }
        else
        {
            pd3dImmediateContext->CopyResource(pBackBufferResource, g_pMainRenderTargetTexture);
        }
        pBackBufferResource->Release();
    }
    else
    // Linked list method
    if (g_nOITMethod==OITMETHOD_LINKEDLIST)
    {
#if NUMBER_OF_TILES_X>1 || NUMBER_OF_TILES_Y>1
        // If using MSAA resolve during sort in combination with tiles then we have 
        // to resolve the MSAA background onto the resolved RT first 
        // otherwise regions not covered by tiles will not get background content
        if (g_bMSAAResolveDuringSort && g_MSAASampleDesc.Count>1)
        {
            pd3dImmediateContext->ResolveSubresource(g_pMainRenderTargetTextureResolved, 0, g_pMainRenderTargetTexture, 0, 
                                                     DXGI_FORMAT_R8G8B8A8_UNORM_SRGB);
        }
#endif

        // Save current scissor
        UINT nScissorRect = 1;
        D3D11_RECT CurrentScissorRect;
        pd3dImmediateContext->RSGetScissorRects(&nScissorRect, &CurrentScissorRect);

        // Tile size
        UINT uTileWidth  = (UINT)( ( g_fScreenWidth  + (NUMBER_OF_TILES_X-1) ) / NUMBER_OF_TILES_X );
        UINT uTileHeight = (UINT)( ( g_fScreenHeight + (NUMBER_OF_TILES_Y-1) ) / NUMBER_OF_TILES_Y );

        // Find maximum extent of translucent geometry in 2D screen coordinates
        D3DXMATRIX mModelWVP = mWorldViewProjection;
        GetTransformedBoundingBoxExtents(&g_TransparentMesh[g_nCurrentModel], &mModelWVP, g_fScreenWidth, g_fScreenHeight, 
                                         &g_ModelBoundingBoxExtent);

        // Loop through all tiles
        g_nNumTilesVisible = 0;
        for (int nTileStartPositionX = g_ModelBoundingBoxExtent.left; nTileStartPositionX < g_ModelBoundingBoxExtent.right; nTileStartPositionX += uTileWidth)
        {
            int nTileEndPositionX = min(nTileStartPositionX + uTileWidth, (UINT)g_fScreenWidth - 1);
            for (int nTileStartPositionY = g_ModelBoundingBoxExtent.top; nTileStartPositionY < g_ModelBoundingBoxExtent.bottom; nTileStartPositionY += uTileHeight)
            {
                int nTileEndPositionY = min(nTileStartPositionY + uTileHeight, (UINT)g_fScreenHeight - 1);

                // Skip tiles that are outside the render area
                // This is only required if the tile size is a fixed size independent of resolution
                //if ( (nTileX*uTileWidth>g_fScreenWidth) || (nTileY*uTileHeight>g_fScreenHeight) ) continue;

                //
                // LINKED LIST Step 1: Store fragments into UAVs
                //

                // Clear start offset buffer to -1
                const UINT dwClearDataMinusOne[1] = { 0xFFFFFFFF };
                pd3dImmediateContext->ClearUnorderedAccessViewUint(g_pStartOffsetBufferUAV, dwClearDataMinusOne);

                // Set scissor rect for current tile
                D3D11_RECT ScissorRect;
                ScissorRect.top    = (LONG)nTileStartPositionY;
                ScissorRect.left   = (LONG)nTileStartPositionX;
                ScissorRect.bottom = (LONG)nTileEndPositionY;
                ScissorRect.right  = (LONG)nTileEndPositionX;
                pd3dImmediateContext->RSSetScissorRects(1, &ScissorRect);

                // Update tile coordinates CB
                D3D11_MAPPED_SUBRESOURCE MappedSubResource;
                pd3dImmediateContext->Map(g_pTileCoordinatesCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedSubResource );
                ((TILE_COORDINATES_CONSTANT_BUFFER_STRUCT *)MappedSubResource.pData)->vRectangleCoordinates = 
                    D3DXVECTOR4( (float)ScissorRect.left, (float)ScissorRect.top, (float)ScissorRect.right, (float)ScissorRect.bottom);
                ((TILE_COORDINATES_CONSTANT_BUFFER_STRUCT *)MappedSubResource.pData)->vTileSize = 
                    D3DXVECTOR4( (float)uTileWidth, (float)uTileHeight, 0, 0);
                pd3dImmediateContext->Unmap(g_pTileCoordinatesCB, 0);
                                    
                // Set render target, depth buffer and UAVs
                pRTV[0] = NULL;
                pUAV[0] = g_pStartOffsetBufferUAV;
                pUAV[1] = g_pFragmentAndLinkStructuredBufferUAV;
                pUAV[2] = NULL;
                pUAV[3] = NULL;
                pd3dImmediateContext->OMSetRenderTargetsAndUnorderedAccessViews(1, pRTV, g_pDepthStencilTextureDSV, 
                                                                                1, 4, pUAV, pUAVCounters );

                // Set shaders
                pd3dImmediateContext->VSSetShader( g_pMainVS, NULL, 0 );
                pd3dImmediateContext->PSSetShader( g_pStoreFragmentsPS_LinkedList, NULL, 0 ); 

                // Set stencil buffer to increment for each fragment
                pd3dImmediateContext->OMSetDepthStencilState(g_pDepthTestEnabledNoDepthWritesStencilWriteIncrementDSS, 0x00);

                // Disable color writes
                pd3dImmediateContext->OMSetBlendState(g_pColorWritesOff, 0, 0xffffffff);

                // Bind the mesh constant buffer at slot 1 for all stages
                pBuffers[0] = g_pTransparentMeshCB;
                pBuffers[1] = g_pTileCoordinatesCB;
                pd3dImmediateContext->VSSetConstantBuffers( 1, 1, pBuffers );
                pd3dImmediateContext->PSSetConstantBuffers( 1, 2, pBuffers );
                pd3dImmediateContext->CSSetConstantBuffers( 1, 2, pBuffers );

                // Calculate off-centered projection matrix corresponding to current tile
                D3DXMATRIX mTileProjection;
                CalculateOffCenteredProjectionMatrixFrom2DRectangle(&mTileProjection, g_fScreenWidth, g_fScreenHeight, 
                                                                    g_Camera.GetNearClip(), g_Camera.GetFarClip(), 
                                                                    D3DX_PI/4, ScissorRect);

                // Calculate combined WVP transformation matrix for current tile
                D3DXMATRIX TransMatrixTile = mWorld * (*g_Camera.GetViewMatrix()) * mTileProjection;
                
                // Set bounding box status: this updates the FrameInfluenceOffset member
                // of all sub-meshes with visibility info
                PerformBoundingBoxCheck(&g_TransparentMesh[g_nCurrentModel], &TransMatrixTile);

                // Debug info
                if (g_TransparentMesh[g_nCurrentModel].GetMesh(0)->FrameInfluenceOffset>0) g_nNumTilesVisible++;

                // Render transparent model with bounding box testing on
                RenderModel(pd3dImmediateContext, &g_TransparentMesh[g_nCurrentModel], true);


                //
                // LINKED LIST Step 2: Sorting and displaying pass
                //

                if (g_bMSAAResolveDuringSort && g_MSAASampleDesc.Count>1)
                {
                    // Set render target to back buffer directly since we will also resolve on the fly
                    //pRTV[0] = DXUTGetD3D11RenderTargetView();
                    
                    // Set render target to resolved render target since we will also resolve on the fly
                    pRTV[0] = g_pMainRenderTargetTextureResolvedRTV;
                    pUAV[0] = NULL;
                    pUAV[1] = NULL;
                    pUAV[2] = NULL;
                    pUAV[3] = NULL;
                    pd3dImmediateContext->OMSetRenderTargetsAndUnorderedAccessViews(1, pRTV, NULL, 1, 4, 
                                                                                    pUAV, pUAVCounters);

                    // No depthstencil test since no depth buffer is bound
                    pd3dImmediateContext->OMSetDepthStencilState(g_pDepthTestDisabledDSS, 0x00);
                    
                    // Set shaders
                    pd3dImmediateContext->VSSetShader( g_pVSPassThrough, NULL, 0 );
                    pd3dImmediateContext->PSSetShader( g_pRenderFragmentsWithResolvePS_LinkedList, NULL, 0 ); 

                    // Set shader resources
                    pSRV[0] = g_pStartOffsetBufferSRV;
                    pSRV[1] = g_pFragmentAndLinkStructuredBufferSRV;
                    pSRV[2] = NULL;
                    pSRV[3] = g_pCopyOfMainRenderTargetTextureSRV;
                    pd3dImmediateContext->PSSetShaderResources(0, 4, pSRV);

                    // Disable blending (background color comes from a copy 
                    // of the main render target bound as texture)
                    pd3dImmediateContext->OMSetBlendState(g_pBlendStateNoBlend, 0, 0xffffffff);
                    
                    // Set up fullscreen quad rendering
                    stride = sizeof( SIMPLEVERTEX );
                    offset = 0;
                    pd3dImmediateContext->IASetVertexBuffers( 0, 1, &g_pFullscreenQuadVB, &stride, &offset );
                    pd3dImmediateContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
                    pd3dImmediateContext->IASetInputLayout( g_pSimpleVertexLayout );

                    // Draw fullscreen quad
                    pd3dImmediateContext->Draw( 4, 0);
                }
                else
                {
                    // Set render target, depth buffer and UAVs (no UAVs are used during the resolve pass)
                    pRTV[0] = g_pMainRenderTargetTextureRTV;
                    pUAV[0] = NULL;
                    pUAV[1] = NULL;
                    pUAV[2] = NULL;
                    pUAV[3] = NULL;
                    pd3dImmediateContext->OMSetRenderTargetsAndUnorderedAccessViews(1, pRTV, g_pDepthStencilTextureDSV, 
                                                                                    1, 4, pUAV, pUAVCounters);

                    // Set stencil pass to pass if stencil value is above 0
                    pd3dImmediateContext->OMSetDepthStencilState(g_pDepthTestDisabledStencilTestLessDSS, 0x00);

                    // Set shaders
                    pd3dImmediateContext->VSSetShader( g_pVSPassThrough, NULL, 0 );
                    pd3dImmediateContext->PSSetShader( g_pRenderFragmentsPS_LinkedList, NULL, 0 ); 

                    // Set shader resources
                    pSRV[0] = g_pStartOffsetBufferSRV;
                    pSRV[1] = g_pFragmentAndLinkStructuredBufferSRV;
                    pSRV[2] = NULL;
                    pSRV[3] = g_pCopyOfMainRenderTargetTextureSRV;
                    pd3dImmediateContext->PSSetShaderResources(0, 4, pSRV);

                    // Disable blending (background color comes from a copy of the back buffer bound as texture)
                    pd3dImmediateContext->OMSetBlendState(g_pBlendStateNoBlend, 0, 0xffffffff);
                    
                    // Blend sorted layers with the background
                    //pd3dImmediateContext->OMSetBlendState(g_pBlendStateBackgroundUnderBlending, 0, 0xffffffff);

                    // Set up fullscreen quad rendering
                    stride = sizeof( SIMPLEVERTEX );
                    offset = 0;
                    pd3dImmediateContext->IASetVertexBuffers( 0, 1, &g_pFullscreenQuadVB, &stride, &offset );
                    pd3dImmediateContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
                    pd3dImmediateContext->IASetInputLayout( g_pSimpleVertexLayout );

                    // Draw fullscreen quad
                    pd3dImmediateContext->Draw( 4, 0);
                }

                // To avoid debug error messages
                pSRV[0] = NULL;
                pSRV[1] = NULL;
                pSRV[2] = NULL;
                pd3dImmediateContext->PSSetShaderResources(0, 3, pSRV);
            }
        }

        // Restore scissor rect
        pd3dImmediateContext->RSSetScissorRects(1, &CurrentScissorRect);

        // If rendering was done on a render target then ensure it gets resolved/copied onto the back buffer
        if (g_bMSAAResolveDuringSort && g_MSAASampleDesc.Count>1)
        {
            // Copy onto back buffer
            ID3D11Resource* pBackBufferResource;
            DXUTGetD3D11RenderTargetView()->GetResource(&pBackBufferResource);
            pd3dImmediateContext->CopyResource(pBackBufferResource, g_pMainRenderTargetTextureResolved);
            pBackBufferResource->Release();
        }
        else
        {
            // Resolve/copy onto back buffer
            ID3D11Resource* pBackBufferResource;
            DXUTGetD3D11RenderTargetView()->GetResource(&pBackBufferResource);
            if (g_MSAASampleDesc.Count>1)
            {
                pd3dImmediateContext->ResolveSubresource(pBackBufferResource, 0, g_pMainRenderTargetTexture, 0, 
                                                         DXGI_FORMAT_R8G8B8A8_UNORM_SRGB);
            }
            else
            {
                pd3dImmediateContext->CopyResource(pBackBufferResource, g_pMainRenderTargetTexture);
            }
            pBackBufferResource->Release();
        }
    }

    // Set shader resources
    pSRV[0] = NULL;
    pSRV[1] = NULL;
    pSRV[2] = NULL;
    pSRV[3] = NULL;
    pd3dImmediateContext->PSSetShaderResources(0, 4, pSRV);

    // Set rendering to back buffer and disable UAVs
    pRTV[0] = DXUTGetD3D11RenderTargetView();
    pUAV[0] = NULL;
    pUAV[1] = NULL;
    pUAV[2] = NULL;
    pd3dImmediateContext->OMSetRenderTargetsAndUnorderedAccessViews(1, pRTV, NULL, 
                                                                    1, 3, pUAV, pUAVCounters);

    // Set depth stencil state
    pd3dImmediateContext->OMSetDepthStencilState(g_pDepthTestDisabledDSS, 0x00);

    // Render the HUD
    if (g_nRenderHUD>0)
    {
        DXUT_BeginPerfEvent( DXUT_PERFEVENTCOLOR, L"HUD / Stats" );
        if (g_nRenderHUD>1)
        {
            g_HUD.OnRender( fElapsedTime );
            g_SampleUI.OnRender( fElapsedTime );
        }
        RenderText();
        DXUT_EndPerfEvent();
    }

    // Check if current frame needs to be dumped to disk
    if (dwFrameNumber==g_dwFrameNumberToDump)
    {
        // Retrieve RT resource
        ID3D11Resource *pRTResource;
        DXUTGetD3D11RenderTargetView()->GetResource(&pRTResource);

        // Retrieve a Texture2D interface from resource
        ID3D11Texture2D* RTTexture;
        hr = pRTResource->QueryInterface( __uuidof( ID3D11Texture2D ), ( LPVOID* )&RTTexture);

        // Check if RT is multisampled or not
        D3D11_TEXTURE2D_DESC    TexDesc;
        RTTexture->GetDesc(&TexDesc);
        if (TexDesc.SampleDesc.Count>1)
        {
            // RT is multisampled, need resolving before dumping to disk

            // Create single-sample RT of the same type and dimensions
            DXGI_SAMPLE_DESC SingleSample = { 1, 0 };
            TexDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            TexDesc.MipLevels = 1;
            TexDesc.Usage = D3D11_USAGE_DEFAULT;
            TexDesc.CPUAccessFlags = 0;
            TexDesc.BindFlags = 0;
            TexDesc.SampleDesc = SingleSample;

            ID3D11Texture2D *pSingleSampleTexture;
            hr = pd3dDevice->CreateTexture2D(&TexDesc, NULL, &pSingleSampleTexture);

            pd3dImmediateContext->ResolveSubresource(pSingleSampleTexture, 0, RTTexture, 0, TexDesc.Format);

            // Copy RT into STAGING texture
            pd3dImmediateContext->CopyResource(g_pCopyOfMainRenderTargetTextureSTAGING, pSingleSampleTexture);

            hr = D3DX11SaveTextureToFile(pd3dImmediateContext, g_pCopyOfMainRenderTargetTextureSTAGING, 
                                         D3DX11_IFF_BMP, L"RTOutput.bmp");

            SAFE_RELEASE(pSingleSampleTexture);
            
        }
        else
        {
            // Single sample case

            // Copy RT into STAGING texture
            pd3dImmediateContext->CopyResource(g_pCopyOfMainRenderTargetTextureSTAGING, pRTResource);

            hr = D3DX11SaveTextureToFile(pd3dImmediateContext, g_pCopyOfMainRenderTargetTextureSTAGING, 
                                         D3DX11_IFF_BMP, L"RTOutput.bmp");
        }

        SAFE_RELEASE(RTTexture);
        SAFE_RELEASE(pRTResource);
    }

    // Increase frame number
    dwFrameNumber++;
}

//--------------------------------------------------------------------------------------
// Release D3D11 resources created in OnD3D11ResizedSwapChain 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11ReleasingSwapChain( void* pUserContext )
{
    DestroyResources();
    g_DialogResourceManager.OnD3D11ReleasingSwapChain();
}


//--------------------------------------------------------------------------------------
// Release D3D11 resources created in OnD3D11CreateDevice 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11DestroyDevice( void* pUserContext )
{
    g_DialogResourceManager.OnD3D11DestroyDevice();
    g_D3DSettingsDlg.OnD3D11DestroyDevice();
    DXUTGetGlobalResourceCache().OnDestroyDevice();
    SAFE_DELETE( g_pTxtHelper );

    // Release queries
    SAFE_RELEASE(g_pEventQuery);

    // Release transparent meshes
    for ( int iMeshType=0; iMeshType<NUM_TRANSPARENT_MESHES; iMeshType++ )
    {
        g_TransparentMesh[iMeshType].Destroy();
    }
    g_OpaqueMesh.Destroy();

    // Texture
    SAFE_RELEASE( g_pDiffuseTextureRV );
    SAFE_RELEASE( g_pWhiteTextureRV );

    // Geometry
    SAFE_RELEASE( g_pCubeIB );
    SAFE_RELEASE( g_pCubeVB );
    SAFE_RELEASE( g_pFullscreenQuadVB );

    // Input layouts
    SAFE_RELEASE( g_pExtendedVertexLayout );
    SAFE_RELEASE( g_pSimpleVertexLayout );

    // Shaders
    SAFE_RELEASE( g_pLightingAndTexturingPS );
    SAFE_RELEASE( g_pMainVS );
    SAFE_RELEASE( g_pVSPassThrough );

    // Constant buffers
    SAFE_RELEASE( g_pMainCB);
    for (int i=0; i<MAX_NUMBER_OF_CUBES; i++)
    {
        SAFE_RELEASE(g_pCubeCB[i]);
    }
    SAFE_RELEASE( g_pOpaqueMeshCB );
    SAFE_RELEASE( g_pTransparentMeshCB );
    SAFE_RELEASE( g_pTileCoordinatesCB );

    // State objects
    SAFE_RELEASE( g_pDepthTestDisabledStencilTestLessDSS );
    SAFE_RELEASE( g_pDepthTestEnabledNoDepthWritesStencilWriteIncrementDSS );
    SAFE_RELEASE( g_pDepthTestEnabledNoDepthWritesDSS );
    SAFE_RELEASE( g_pDepthTestEnabledDSS );
    SAFE_RELEASE( g_pDepthTestDisabledDSS );
    SAFE_RELEASE( g_pColorWritesOff );
    //SAFE_RELEASE( g_pBlendStateBackgroundUnderBlending );
    SAFE_RELEASE( g_pBlendStateSrcAlphaInvSrcAlphaBlend );
    SAFE_RELEASE( g_pBlendStateNoBlend );
    SAFE_RELEASE( g_pRasterizerStateSolid );
    SAFE_RELEASE( g_pRasterizerStateWireframe );
    SAFE_RELEASE( g_pSamplerStatePointClamp );
    SAFE_RELEASE( g_pSamplerStateLinearWrap );
    SAFE_RELEASE( g_pSamplerStateLinearClamp );
}


//--------------------------------------------------------------------------------------
// RenderCubes()
//--------------------------------------------------------------------------------------
void RenderCubes(ID3D11DeviceContext* pd3dContext)
{
    // Set up vertex and index buffers
    UINT stride = sizeof( EXTENDEDVERTEX );
    UINT offset = 0;
    pd3dContext->IASetVertexBuffers( 0, 1, &g_pCubeVB, &stride, &offset );
    pd3dContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    pd3dContext->IASetInputLayout( g_pExtendedVertexLayout );
    pd3dContext->IASetIndexBuffer( g_pCubeIB, DXGI_FORMAT_R16_UINT, 0 );

    // Loop through all cubes
    // Cube rendering should really be instanced but there are so few that 
    // it doesn't really matter for an SDK sample
    for (int i=0; i<g_nNumCubes; i++)
    {
        // Bind the mesh constant buffer at slot 1 for all stages
        ID3D11Buffer* pBuffers[1] = { g_pCubeCB[i] };
        pd3dContext->VSSetConstantBuffers( 1, 1, pBuffers );
        pd3dContext->PSSetConstantBuffers( 1, 1, pBuffers );

        // Draw cube
        pd3dContext->DrawIndexed( 36, 0, 0);
    }
}


//--------------------------------------------------------------------------------------
// RenderModel()
//--------------------------------------------------------------------------------------
void RenderModel(ID3D11DeviceContext* pd3dContext, CDXUTSDKMesh* pDXUTMesh, bool bUseBoundingBoxTesting)
{
#define MAX_D3D11_VERTEX_STREAMS D3D11_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT

    // Set input layut
    pd3dContext->IASetInputLayout( g_pExtendedVertexLayout );

    // Loop through all meshes in model
    for (UINT mesh = 0; mesh < pDXUTMesh->GetNumMeshes(); ++mesh )
    {
        SDKMESH_MESH* pMesh = pDXUTMesh->GetMesh( mesh );

        // If bounding box testing is enabled then skip mesh if visibility status (stored in FrameInfluenceOffset) is 0
        if (bUseBoundingBoxTesting)
        {
            if (pMesh->FrameInfluenceOffset==0) continue;
        }

        UINT Strides[MAX_D3D11_VERTEX_STREAMS];
        UINT Offsets[MAX_D3D11_VERTEX_STREAMS];
        ID3D11Buffer* pVB[MAX_D3D11_VERTEX_STREAMS];
        
        if( pMesh->NumVertexBuffers > MAX_D3D11_VERTEX_STREAMS )
        {
            return;
        }
            
        // Set vertex buffers
        for( UINT i = 0; i < pMesh->NumVertexBuffers; i++ )
        {
            pVB[i] = pDXUTMesh->GetVB11( mesh, i );
            Strides[i] = pDXUTMesh->GetVertexStride( mesh, i );
            Offsets[i] = 0;
        }
        pd3dContext->IASetVertexBuffers( 0, pMesh->NumVertexBuffers, pVB, Strides, Offsets );

        // Set index buffer
        ID3D11Buffer* pIB;
        pIB = pDXUTMesh->GetIB11( mesh );
        pd3dContext->IASetIndexBuffer( pDXUTMesh->GetIB11( mesh ), pDXUTMesh->GetIBFormat11( mesh ), 0 );

        // Loop through all subsets in mesh
        for( UINT subset = 0; subset < pDXUTMesh->GetNumSubsets( mesh ); ++subset )
        {
            SDKMESH_SUBSET* pSubset = NULL;
            D3D11_PRIMITIVE_TOPOLOGY PrimType;

            // Get the subset
            pSubset = pDXUTMesh->GetSubset( mesh, subset );

            // Set primitive topology
            PrimType = CDXUTSDKMesh::GetPrimitiveType11( ( SDKMESH_PRIMITIVE_TYPE )pSubset->PrimitiveType );
            pd3dContext->IASetPrimitiveTopology( PrimType );

            // Set diffuse texture
            SDKMESH_MATERIAL* pMat = pDXUTMesh->GetMaterial( pSubset->MaterialID );

            // If no diffuse texture is present in the model then use a default white texture
            if (pMat->pDiffuseRV11==NULL)
            {
                pd3dContext->PSSetShaderResources( 0, 1, &g_pWhiteTextureRV );
            }
            else
            if ( !IsErrorResource( pMat->pDiffuseRV11 ) )
            {
                pd3dContext->PSSetShaderResources( 0, 1, &pMat->pDiffuseRV11 );
            }

            // Draw
            pd3dContext->DrawIndexed( ( UINT )pSubset->IndexCount, ( UINT )pSubset->IndexStart, 
                                      ( UINT )pSubset->VertexStart );
        }
    }
}

//--------------------------------------------------------------------------------------
// InitializeCubeList()
//--------------------------------------------------------------------------------------
void InitializeCubeList()
{
    srand(g_uRandSeed);
    for (int i=0; i<MAX_NUMBER_OF_CUBES; i++)
    {
        OBJECT_STRUCT ObjectElement;

        // Add new element in list
        ObjectElement.vPosition    = D3DXVECTOR3( FLOAT_RANDOM(1.0f), FLOAT_RANDOM(0.5f) - 1.5f, FLOAT_RANDOM(1.0f) ); 
        ObjectElement.vOrientation = D3DXVECTOR4( FLOAT_RANDOM(1.0f), FLOAT_RANDOM(1.0f), FLOAT_RANDOM(1.0f), FLOAT_POSITIVE_RANDOM(PI) );
        ObjectElement.fScale       = FLOAT_POSITIVE_RANDOM(0.2f) + 0.1f;
        float fGrayscale           = FLOAT_POSITIVE_RANDOM(0.7f) + 0.2f;
        ObjectElement.vColor       = D3DXVECTOR4(fGrayscale, fGrayscale, fGrayscale, 1.0f);

        // Add new element in list
        g_pCubeList[i] = ObjectElement;
    }
}

//--------------------------------------------------------------------------------------
// UpdateCubeConstantBuffers()
//--------------------------------------------------------------------------------------
void UpdateCubeConstantBuffers(ID3D11DeviceContext* pd3dContext)
{
    // Loop through all spheres
    for (int i=0; i<MAX_NUMBER_OF_CUBES; i++)
    {
        // Update world matrix
        D3DXMATRIX mScaleMatrix;
        float fCubeScale = g_pCubeList[i].fScale;
        D3DXMatrixScaling(&mScaleMatrix, fCubeScale, fCubeScale, fCubeScale);
        D3DXMATRIX mRotationMatrix;
        D3DXVECTOR3 CubeRotationVector(g_pCubeList[i].vOrientation.x, 
                                       g_pCubeList[i].vOrientation.y, 
                                       g_pCubeList[i].vOrientation.z);
        D3DXMatrixRotationAxis(&mRotationMatrix, &CubeRotationVector, g_pCubeList[i].vOrientation.w);
        D3DXMATRIX mTranslation;
        D3DXMatrixTranslation(&mTranslation, g_pCubeList[i].vPosition.x, 
                                             g_pCubeList[i].vPosition.y, 
                                             g_pCubeList[i].vPosition.z);
        D3DXMATRIX mWorld;
        mWorld = mScaleMatrix * mRotationMatrix * mTranslation;
        
        // Transpose matrices before passing to VS
        D3DXMATRIX mTransposedWorld;
        D3DXMatrixTranspose(&mTransposedWorld, &mWorld);    

        D3DXMATRIX mWorldViewProjection;
        mWorldViewProjection = mWorld * ( (*g_Camera.GetViewMatrix()) * (*g_Camera.GetProjMatrix()) );
        D3DXMATRIX mTransposedWorldViewProjection;
        D3DXMatrixTranspose(&mTransposedWorldViewProjection, &mWorldViewProjection);    

        // Update constant buffer
        D3D11_MAPPED_SUBRESOURCE MappedSubResource;
        pd3dContext->Map(g_pCubeCB[i], 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedSubResource );
        ((PER_MESH_CONSTANT_BUFFER_STRUCT *)MappedSubResource.pData)->mWorld                = mTransposedWorld;
        ((PER_MESH_CONSTANT_BUFFER_STRUCT *)MappedSubResource.pData)->mWorldViewProjection  = mTransposedWorldViewProjection;
        ((PER_MESH_CONSTANT_BUFFER_STRUCT *)MappedSubResource.pData)->vMeshColor = g_pCubeList[i].vColor;
        pd3dContext->Unmap(g_pCubeCB[i], 0);
    }
}


//--------------------------------------------------------------------------------------
// Transform model's bounding box by supplied transformation matrix and apply perspective
// divide as well as viewport mapping to retrieve 2D screen coordinate results.
// The maximum extent of the bounding box is then returned
//--------------------------------------------------------------------------------------
void GetTransformedBoundingBoxExtents(CDXUTSDKMesh* pSDKMesh, D3DXMATRIX* pTransformationMatrix, float fRenderWidth, float fRenderHeight, 
                                      RECT *pBoundingBoxExtent)
{
    float fMinX = fRenderWidth-1.0f;
    float fMaxX = 0.0f;
    float fMinY = fRenderHeight-1.0f;
    float fMaxY = 0.0f;
    bool  bPartiallyIntersectingFrontClipPlane = false;

    D3DXVECTOR3 vUnitCube[8] = { D3DXVECTOR3( 1.0,  1.0, 1.0),  D3DXVECTOR3( 1.0,  1.0, -1.0 ), 
                                 D3DXVECTOR3( 1.0, -1.0, 1.0 ), D3DXVECTOR3( 1.0, -1.0, -1.0 ),
                                 D3DXVECTOR3(-1.0,  1.0, 1.0),  D3DXVECTOR3(-1.0,  1.0, -1.0 ), 
                                 D3DXVECTOR3(-1.0, -1.0, 1.0 ), D3DXVECTOR3(-1.0, -1.0, -1.0 ) };

    // Loop through all meshes in SDKMesh                
    for (DWORD k=0; k<pSDKMesh->GetNumMeshes(); k++)
    {
        for (int p=0; p<8; p++)
        {
            BOUNDING_BOX_STRUCT MeshBoundingBox;

            // Prepare Bounding box structure for specified mesh    
            MeshBoundingBox.vPoint[p].x = pSDKMesh->GetMeshBBoxCenter(k).x + 
                                          vUnitCube[p].x*pSDKMesh->GetMeshBBoxExtents(k).x;
            MeshBoundingBox.vPoint[p].y = pSDKMesh->GetMeshBBoxCenter(k).y + 
                                          vUnitCube[p].y*pSDKMesh->GetMeshBBoxExtents(k).y;
            MeshBoundingBox.vPoint[p].z = pSDKMesh->GetMeshBBoxCenter(k).z + 
                                          vUnitCube[p].z*pSDKMesh->GetMeshBBoxExtents(k).z;

            // Transform each bounding box point with specified transformation matrix
            D3DXVECTOR4 TransformedPoint;
            D3DXVec3Transform(&TransformedPoint, &MeshBoundingBox.vPoint[p], pTransformationMatrix);

            // Only include points that are in front of the camera
            if (TransformedPoint.w>0)
            {
                // Apply perspective divide to obtain normalized device coordinates
                TransformedPoint.x /= TransformedPoint.w;
                TransformedPoint.y /= TransformedPoint.w;
                TransformedPoint.z /= TransformedPoint.w;

                // Apply viewport mapping
                TransformedPoint.x = 0.5f * (TransformedPoint.x + 1.0f) * fRenderWidth;
                TransformedPoint.y = 0.5f * (1.0f - TransformedPoint.y) * fRenderHeight;

                // Update mesh extents
                if (TransformedPoint.x<fMinX) fMinX = TransformedPoint.x;
                if (TransformedPoint.x>fMaxX) fMaxX = TransformedPoint.x;
                if (TransformedPoint.y<fMinY) fMinY = TransformedPoint.y;
                if (TransformedPoint.y>fMaxY) fMaxY = TransformedPoint.y;
            }
            else
            {
                bPartiallyIntersectingFrontClipPlane = true;
            }
        }
    }

    // Special case to make bounding box extent cover the whole screen if the bounding 
    // box at least partially covers the front clip plane
    if (bPartiallyIntersectingFrontClipPlane)
    {
        fMinX = 0.0f;
        fMaxX = fRenderWidth-1.0f;
        fMinY = 0.0f;
        fMaxY = fRenderHeight-1.0f;
    }


    // Return extents
    pBoundingBoxExtent->left   = (LONG)max(fMinX, 0);
    pBoundingBoxExtent->right  = (LONG)min(fMaxX, fRenderWidth-1.0f);
    pBoundingBoxExtent->top    = (LONG)max(fMinY, 0);
    pBoundingBoxExtent->bottom = (LONG)min(fMaxY, fRenderHeight-1.0f);
}


//--------------------------------------------------------------------------------------
// Bounding box test
// We overload the SDKMESH_MESH.FrameInfluenceOffset variable to indicate bounding
// box visibility.
// FrameInfluenceOffset = 0 : not visible
// FrameInfluenceOffset = 1 : at least partially visible
//--------------------------------------------------------------------------------------
void PerformBoundingBoxCheck(CDXUTSDKMesh* pSDKMesh, D3DXMATRIX* pTransformationMatrix)
{
    // Extract clip planes from specified transformation matrix
    D3DXVECTOR4 pTileFrustumPlaneEquation[6];
    ExtractPlanesFromFrustum( pTileFrustumPlaneEquation, pTransformationMatrix );

    // Loop through all meshes in SDKMesh                
    for (DWORD k=0; k<pSDKMesh->GetNumMeshes(); k++)
    {
        BOUNDING_BOX_STRUCT MeshBoundingBox;
        D3DXVECTOR3 vUnitCube[8] = { D3DXVECTOR3( 1.0,  1.0, 1.0),  D3DXVECTOR3( 1.0,  1.0, -1.0 ), 
                                     D3DXVECTOR3( 1.0, -1.0, 1.0 ), D3DXVECTOR3( 1.0, -1.0, -1.0 ),
                                     D3DXVECTOR3(-1.0,  1.0, 1.0),  D3DXVECTOR3(-1.0,  1.0, -1.0 ), 
                                     D3DXVECTOR3(-1.0, -1.0, 1.0 ), D3DXVECTOR3(-1.0, -1.0, -1.0 ) };

        // Prepare Bounding box structure for specified mesh    
        for (int p=0; p<8; p++)
        {
            MeshBoundingBox.vPoint[p].x = pSDKMesh->GetMeshBBoxCenter(k).x + 
                                          vUnitCube[p].x*pSDKMesh->GetMeshBBoxExtents(k).x;
            MeshBoundingBox.vPoint[p].y = pSDKMesh->GetMeshBBoxCenter(k).y + 
                                          vUnitCube[p].y*pSDKMesh->GetMeshBBoxExtents(k).y;
            MeshBoundingBox.vPoint[p].z = pSDKMesh->GetMeshBBoxCenter(k).z + 
                                          vUnitCube[p].z*pSDKMesh->GetMeshBBoxExtents(k).z;
        }

        // Determine if bounding box is contained within specified clip planes
        pSDKMesh->GetMesh(k)->FrameInfluenceOffset = 
            AxisAlignedBoundingBoxWithinFrustum(&MeshBoundingBox, pTileFrustumPlaneEquation)>=0.0f ? 1 : 0;
    }
}

//--------------------------------------------------------------------------------------
// Helper function for command line retrieval
//--------------------------------------------------------------------------------------
bool IsNextArg( WCHAR*& strCmdLine, WCHAR* strArg )
{
   int nArgLen = (int) wcslen(strArg);
   int nCmdLen = (int) wcslen(strCmdLine);

   if( nCmdLen >= nArgLen && 
      _wcsnicmp( strCmdLine, strArg, nArgLen ) == 0 && 
      (strCmdLine[nArgLen] == 0 || strCmdLine[nArgLen] == L':' || strCmdLine[nArgLen] == L'=' ) )
   {
      strCmdLine += nArgLen;
      return true;
   }

   return false;
}


//--------------------------------------------------------------------------------------
// Helper function for command line retrieval.  Updates strCmdLine and strFlag 
//      Example: if strCmdLine=="-width:1024 -forceref"
// then after: strCmdLine==" -forceref" and strFlag=="1024"
//--------------------------------------------------------------------------------------
bool GetCmdParam( WCHAR*& strCmdLine, WCHAR* strFlag )
{
   if( *strCmdLine == L':' || *strCmdLine == L'=' )
   {       
      strCmdLine++; // Skip ':'

      // Place NULL terminator in strFlag after current token
      wcscpy_s( strFlag, 256, strCmdLine );
      WCHAR* strSpace = strFlag;
      while (*strSpace && (*strSpace > L' '))
         strSpace++;
      *strSpace = 0;

      // Update strCmdLine
      strCmdLine += wcslen(strFlag);
      return true;
   }
   else
   {
      strFlag[0] = 0;
      return false;
   }
}

//--------------------------------------------------------------------------------------
// CreateStagingFromTexture2D
//--------------------------------------------------------------------------------------
void CreateStagingFromTexture2D(ID3D11Device *pDev, ID3D11Texture2D* pTex, ID3D11Texture2D** ppTexSTAGING)
{
    D3D11_TEXTURE2D_DESC    Tex2DDesc;
    HRESULT                    hr;

    pTex->GetDesc(&Tex2DDesc);
    Tex2DDesc.Usage = D3D11_USAGE_STAGING;
    Tex2DDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
    Tex2DDesc.BindFlags = 0;

    hr = pDev->CreateTexture2D(&Tex2DDesc, NULL, ppTexSTAGING);
}


//--------------------------------------------------------------------------------------
// CreateStagingFromBuffer
//--------------------------------------------------------------------------------------
void CreateStagingFromBuffer(ID3D11Device *pDev, ID3D11Buffer* pBuffer, ID3D11Buffer** ppBufferSTAGING)
{
    D3D11_BUFFER_DESC    BufferDesc;
    HRESULT                hr;

    pBuffer->GetDesc(&BufferDesc);
    BufferDesc.Usage = D3D11_USAGE_STAGING;
    BufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
    BufferDesc.BindFlags = 0;
    BufferDesc.MiscFlags = 0;

    hr = pDev->CreateBuffer(&BufferDesc, NULL, ppBufferSTAGING);
}


//--------------------------------------------------------------------------------------
// Calculate off-centered projection matrix corresponding to 2D screen coordinate 
// rectangle, using the specified fullscreen projection matrix parameters
// This functionality is useful when manually tiling the screen and performing bounding
// box detection of objects with those tiles.
//--------------------------------------------------------------------------------------
void CalculateOffCenteredProjectionMatrixFrom2DRectangle(D3DXMATRIX* pOffCenteredProjectionMatrix, 
                                                         float fFullscreenWidth, float fFullscreenHeight, 
                                                         float fFulscreenZNear, float fFullscreenZFar, 
                                                         float fFullscreenVerticalFOV, RECT RectArea)
{
    // Calculate parameters for offscreen projection matrix corresponding 
    // to specified 2D projection on the screen
    float fFullscreenAspectRatio = fFullscreenWidth/fFullscreenHeight;

    float h = 2.0f * fFulscreenZNear * tan(0.5f*fFullscreenVerticalFOV);
    float w = h * fFullscreenAspectRatio;

    float l = -0.5f * w + (w*RectArea.left)   / fFullscreenWidth;
    float r = -0.5f * w + (w*RectArea.right)  / fFullscreenWidth;
    float t =  0.5f * h - (h*RectArea.top)    / fFullscreenHeight;
    float b =  0.5f * h - (h*RectArea.bottom) / fFullscreenHeight;
    
    D3DXMatrixPerspectiveOffCenterLH(pOffCenteredProjectionMatrix, l, r, b, t, fFulscreenZNear, fFullscreenZFar);
}


//--------------------------------------------------------------------------------------
// Helper function to calculate the distance of a point from a plane
//--------------------------------------------------------------------------------------
float DistanceToPoint( const D3DXVECTOR4 *pPlaneEquation, const D3DXVECTOR3* pPoint )
{
    return ( pPlaneEquation->x*pPoint->x + pPlaneEquation->y*pPoint->y + pPlaneEquation->z*pPoint->z + pPlaneEquation->w );
}


//--------------------------------------------------------------------------------------
// Determine if a point is contained within a frustum
// pFrustumPlaneEquations is an array of 6 D3DXVECTOR4 defining the plane equations of
// the frustum.
// Returns  1.0f if AA bounding box is fully contained within frustum
//          0.0f if AA bounding box is partially contained within frustum
//         -1.0f if AA bounding box is fully outside the frustum
//--------------------------------------------------------------------------------------
float AxisAlignedBoundingBoxWithinFrustum( BOUNDING_BOX_STRUCT* pBoundingBox, D3DXVECTOR4* pFrustumPlaneEquations )
{
    int nTotalIn = 0;

    // Loop through all 6 plane equation
    for ( int i=0; i<6; i++)
    {
        int nPtIn = 1;
        int nInCount = 8;

        // Loop through all 8 points defining the axis-aligned bounding box
        for ( int j=0; j<8; j++ )
        {
            // Check distance of current point with current plane equation
            if ( DistanceToPoint( &pFrustumPlaneEquations[i], &pBoundingBox->vPoint[j] ) < 0.0f ) 
            {
                // Negative distance means outside the plane
                nInCount--;
                nPtIn = 0;
            }
        }

        // Were all the points outside of plane p?
        if ( nInCount == 0 )
        {
            return -1.0f;   // Bounding box is outside frustum
        }

        // Check if they were all on the "in" side of the plane
        nTotalIn += nPtIn;
    }
    
    // If iTotalIn is 6, then all are inside the view
    if (nTotalIn == 6)
    {
        // Bounding box is fully contained inside frustum
        return 1.0f;
    }

    // Partial intersection
    return 0.0f;
}


//--------------------------------------------------------------------------------------
// Helper function to normalize a plane
//--------------------------------------------------------------------------------------
void NormalizePlane( D3DXVECTOR4* pPlaneEquation )
{
    float mag;
    
    mag = sqrt( pPlaneEquation->x * pPlaneEquation->x + 
                pPlaneEquation->y * pPlaneEquation->y + 
                pPlaneEquation->z * pPlaneEquation->z );
    
    pPlaneEquation->x = pPlaneEquation->x / mag;
    pPlaneEquation->y = pPlaneEquation->y / mag;
    pPlaneEquation->z = pPlaneEquation->z / mag;
    pPlaneEquation->w = pPlaneEquation->w / mag;
}


//--------------------------------------------------------------------------------------
// Extract all 6 plane equations from frustum denoted by supplied matrix
//--------------------------------------------------------------------------------------
void ExtractPlanesFromFrustum( D3DXVECTOR4* pPlaneEquation, const D3DXMATRIX* pMatrix, bool bNormalize )
{
    // Left clipping plane
    pPlaneEquation[0].x = pMatrix->_14 + pMatrix->_11;
    pPlaneEquation[0].y = pMatrix->_24 + pMatrix->_21;
    pPlaneEquation[0].z = pMatrix->_34 + pMatrix->_31;
    pPlaneEquation[0].w = pMatrix->_44 + pMatrix->_41;
    
    // Right clipping plane
    pPlaneEquation[1].x = pMatrix->_14 - pMatrix->_11;
    pPlaneEquation[1].y = pMatrix->_24 - pMatrix->_21;
    pPlaneEquation[1].z = pMatrix->_34 - pMatrix->_31;
    pPlaneEquation[1].w = pMatrix->_44 - pMatrix->_41;
    
    // Top clipping plane
    pPlaneEquation[2].x = pMatrix->_14 - pMatrix->_12;
    pPlaneEquation[2].y = pMatrix->_24 - pMatrix->_22;
    pPlaneEquation[2].z = pMatrix->_34 - pMatrix->_32;
    pPlaneEquation[2].w = pMatrix->_44 - pMatrix->_42;
    
    // Bottom clipping plane
    pPlaneEquation[3].x = pMatrix->_14 + pMatrix->_12;
    pPlaneEquation[3].y = pMatrix->_24 + pMatrix->_22;
    pPlaneEquation[3].z = pMatrix->_34 + pMatrix->_32;
    pPlaneEquation[3].w = pMatrix->_44 + pMatrix->_42;
    
    // Near clipping plane
    pPlaneEquation[4].x = pMatrix->_13;
    pPlaneEquation[4].y = pMatrix->_23;
    pPlaneEquation[4].z = pMatrix->_33;
    pPlaneEquation[4].w = pMatrix->_43;
    
    // Far clipping plane
    pPlaneEquation[5].x = pMatrix->_14 - pMatrix->_13;
    pPlaneEquation[5].y = pMatrix->_24 - pMatrix->_23;
    pPlaneEquation[5].z = pMatrix->_34 - pMatrix->_33;
    pPlaneEquation[5].w = pMatrix->_44 - pMatrix->_43;
    
    // Normalize the plane equations, if requested
    if ( bNormalize )
    {
        NormalizePlane( &pPlaneEquation[0] );
        NormalizePlane( &pPlaneEquation[1] );
        NormalizePlane( &pPlaneEquation[2] );
        NormalizePlane( &pPlaneEquation[3] );
        NormalizePlane( &pPlaneEquation[4] );
        NormalizePlane( &pPlaneEquation[5] );
    }
}



/*****************************************************************************
 * Function Name  : SaveScene
 * Inputs          : None
 * Globals used   : Yes
 * Returns          : Nothing
 * Description    : Saves scene data
 *****************************************************************************/
void SaveScene()
{
    FILE    *f;
    char    pszName[] = "scenedata.txt";
    
    if ( fopen_s(&f, pszName, "wb") != 0)
    {
        OutputDebugString(L"OIT11LinkedLists: Unable to write scenedata.txt\n");
        return;
    }

    // Camera parameters
    fprintf(f, "Camera Position %f %f %f\nCamera Orientation %f %f %f\n", 
                    g_Camera.GetEyePt()->x, g_Camera.GetEyePt()->y, g_Camera.GetEyePt()->z, 
                    g_Camera.GetLookAtPt()->x, g_Camera.GetLookAtPt()->y, g_Camera.GetLookAtPt()->z);

    // Cube data
    for (int i=0; i<MAX_NUMBER_OF_CUBES; i++)
    {
        fprintf(f, "Position %f %f %f  Orientation %f %f %f %f  Scale %f  Color %f %f %f %f\n", 
            g_pCubeList[i].vPosition.x, g_pCubeList[i].vPosition.y, g_pCubeList[i].vPosition.z, 
            g_pCubeList[i].vOrientation.x, g_pCubeList[i].vOrientation.y, g_pCubeList[i].vOrientation.z, g_pCubeList[i].vOrientation.w,
            g_pCubeList[i].fScale, 
            g_pCubeList[i].vColor.x, g_pCubeList[i].vColor.y, g_pCubeList[i].vColor.z, g_pCubeList[i].vColor.w);
    }

    /* Close file */
    fclose(f);
}


/*****************************************************************************
 * Function Name  : LoadScene
 * Inputs         : None
 * Globals used   : Yes
 * Returns        : true or false
 * Description    : Loads scene data
 *****************************************************************************/
bool LoadScene()
{
    FILE    *f;
    char    pszName[] = "scenedata.txt";

    if ( fopen_s(&f, pszName, "rb") != 0)
    {
        OutputDebugString(L"OIT11LinkedLists: Unable to read scenedata.txt");
        return false;
    }

    // Camera parameters
    D3DXVECTOR3 vEyePt, vEyeTo;
    fscanf_s(f, "Camera Position %f %f %f\nCamera Orientation %f %f %f\n", 
                &vEyePt.x, &vEyePt.y, &vEyePt.z, 
                &vEyeTo.x, &vEyeTo.y, &vEyeTo.z);
    g_Camera.SetViewParams(&vEyePt, &vEyeTo);

    // Cube data
    for (int i=0; i<MAX_NUMBER_OF_CUBES; i++)
    {
        fscanf_s(f, "Position %f %f %f  Orientation %f %f %f %f  Scale %f  Color %f %f %f %f\n", 
            &g_pCubeList[i].vPosition.x, &g_pCubeList[i].vPosition.y, &g_pCubeList[i].vPosition.z, 
            &g_pCubeList[i].vOrientation.x, &g_pCubeList[i].vOrientation.y, &g_pCubeList[i].vOrientation.z, &g_pCubeList[i].vOrientation.w,
            &g_pCubeList[i].fScale, 
            &g_pCubeList[i].vColor.x, &g_pCubeList[i].vColor.y, &g_pCubeList[i].vColor.z, &g_pCubeList[i].vColor.w);
    }

    // Close file
    fclose(f);

    // No problem occured
    return true;
}