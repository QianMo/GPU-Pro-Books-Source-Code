#pragma once
#include "Core.h"
#include "Model.h"
#include "IncludeHandler.h"
#include "DeepShadowMap.h"
#include "Effects.h"
#include "..\Libs\DXUT\DXUT.h"
#include "..\Libs\DXUT\DXUTgui.h"

#define BACKBUFFER_FORMAT DXGI_FORMAT_R8G8B8A8_UNORM_SRGB

#define NEAR_PLANE 1.0f
#define FAR_PLANE 1000.0f
#define NEAR_PLANE_LIGHT 1.0f
#define FAR_PLANE_LIGHT 1000.0f

enum CameraMode
{
	CAMERA_MAIN,
	CAMERA_LIGHT,
};

struct HairEffect
{
	ID3DX11Effect								*hairEffect;
	ID3D11InputLayout							*hairInputLayout, *headInputLayout;
	ID3DX11EffectTechnique						*hairRenderTechnique, *hairRenderDeferedTechnique, *hairRenderAATechnique, *hairRenderDeferedModelTechnique;
	ID3DX11EffectMatrixVariable					*hairEffectWorldViewProj, *hairEffectWorld, *hairEffectWorldViewProjLight;
	ID3DX11EffectVectorVariable					*hairEffectLightPos, *hairEffectCameraPos, *hairEffectColor;
	ID3DX11EffectVectorVariable					*hairEffectFrameBufferDimension;
	ID3DX11EffectShaderResourceVariable			*hairEffectLinkedListBufWPRO, *hairEffectNeighborsBufRO, *hairEffectStartElementBuf;
	ID3DX11EffectShaderResourceVariable			*hairEffectDeferedPosInLight, *hairEffectDeferedPosInWorld, *hairEffectDeferedNormal, *hairEffectDeferedColor;
	ID3DX11EffectShaderResourceVariable			*hairEffectTexture;
	ID3DX11EffectScalarVariable					*hairEffectDimension;
};


class Main;
// Create a Main object
CoreResult CreateMain(HWND hwnd, Main **outMain, bool bFullScreen);

class Main : public ICoreBase
{
friend CoreResult CreateMain(HWND hwnd, Main **outMain, bool bFullScreen);
public:
	// Draw a new Frame
	bool NewFrame(float timeSinceLastFrame, float timeRunning, bool windowHasFocus);

	// Windowsize changed
	CoreResult WindowSizeChanged(int width, int height);

	void SetMouseButton (bool rightMousebutton, bool buttonDown);
	
	// returns true if the message was processed
	bool MsgProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

	void OnGUIEvent(UINT nEvent, int nControlID, CDXUTControl *pControl);

protected:
	// Release everything
	void finalRelease();
	
	CoreResult createDeferedShadingTextures();

	// Initialize everything
	CoreResult init(HWND hwnd, bool bFullScreen = false);
	
	// Process input
	bool processInput(float timeSinceLastFrame, float timeRunning, bool windowHasFocus);
	
	// Animate models
	void animate(float timeSinceLastFrame, float timeRunning);

	// Render models
	void render(float timeSinceLastFrame, float timeRunning);

	void setDXUTFontColor(D3DCOLOR fontCol, CDXUTControl *control);

	Core *core;
	CoreCamera camera;
	CoreCamera lightCamera;

	POINT mousePos;

	bool leftMouseButtonDown;
	bool rightMouseButtonDown;
	bool leftMouseButtonTrigger;

	IDXGISwapChain	*swapChain;
	DeepShadowMap	*dsm;
	HWND			hwnd;
	CameraMode		cameraMode;

	Model										*hairModel, **headModels;
	int											numHeadModels;
	HairEffect									hairEffects[NUM_EFFECTS];			// effects defined in EffectMode
	int											currentEffect;
	
	CoreTexture2D								*deferedPosInLight, *deferedPosInWorld, *deferedNormal, *deferedTempColor, *deferedColor;
	ID3D11RenderTargetView						*deferedPosInLightRTV, *deferedPosInWorldRTV, *deferedNormalRTV, *deferedTempColorRTV, *deferedColorRTV;
	ID3D11ShaderResourceView					*deferedPosInLightSRV, *deferedPosInWorldSRV, *deferedNormalSRV, *deferedTempColorSRV, *deferedColorSRV;
	ID3D11RenderTargetView						**deferedRenderTargets;
	ID3D11Buffer								*bufQuad;
	ID3D11InputLayout							*layoutQuad;

	CoreMatrix4x4								headRotation;

	float										lightHeight, lightAngle;
	bool										orbitMode;

	// GUI stuff
	CDXUTDialogResourceManager					dialogResourceManager;
	CDXUTDialog									guiDialog;
	CDXUTComboBox								*comboBoxShadowMode, *comboBoxCameraControl;
	CDXUTSlider									*sliderHairAlpha;
	CDXUTButton									*buttonRotUp, *buttonRotDown, *buttonRotLeft, *buttonRotRight;
	CDXUTCheckBox								*cbOrbit;
	CDXUTButton									*buttonOrbitLightRotPlus, *buttonOrbitLightRotMinus, *buttonOrbitLightHeightPlus, *buttonOrbitLightHeightMinus;
	CDXUTButton									*buttonHairColor;
};

struct SCREEN_VERTEX
{
    CoreVector3 pos;
    float u, v; 
};

#define IDC_COMBOBOX_SHADOWMODE 0
#define IDC_COMBOBOX_CAMERACONTROL 1
#define IDC_SLIDER_HAIR_ALPHA_TEXT 2
#define IDC_SLIDER_HAIR_ALPHA 3
#define IDC_BUTTON_ROT_UP 4
#define IDC_BUTTON_ROT_DOWN 5
#define IDC_BUTTON_ROT_LEFT 6
#define IDC_BUTTON_ROT_RIGHT 7
#define IDC_ROT_BUTTON_TEXT 8
#define IDC_CHECKBOX_ORBIT 9
#define IDC_BUTTON_ORBIT_LIGHT_ROT_PLUS 10
#define IDC_BUTTON_ORBIT_LIGHT_ROT_MINUS 11
#define IDC_ORBIT_LIGHT_ROT_BUTTON_TEXT 12
#define IDC_BUTTON_ORBIT_LIGHT_HEIGHT_PLUS 13
#define IDC_BUTTON_ORBIT_LIGHT_HEIGHT_MINUS 14
#define IDC_ORBIT_LIGHT_HEIGHT_BUTTON_TEXT 15
#define IDC_BUTTON_HAIR_COLOR 16