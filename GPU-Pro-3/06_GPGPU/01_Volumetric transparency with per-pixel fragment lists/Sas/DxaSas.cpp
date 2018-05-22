#include "DXUT.h"	
#include "DxaSas.h"
#include "SasButton.h"
#include "SasSlider.h"
#include "SasCheckbox.h"
#include <string>
#include <sstream>
#include "SDKmisc.h"

Dxa::Sas::Sas(ID3D11Device* device)
:Base(device)
{
	guiDt = 0.0001;
	hide = false;
}

Dxa::Sas::~Sas()
{
}

class FullScreenFunctor : public SasControl::Button::Functor
{
public:
	void operator()()
	{
		DXUTToggleFullScreen();
	}
};

HRESULT Dxa::Sas::createResources()
{
	loadEffect();

	hud.Init( &dialogResourceManager );
    ui.Init( &dialogResourceManager );
    hud.SetCallback( onGuiEvent, this);
    ui.SetCallback( onGuiEvent, this );

	int nextPosition = 10;
	nextControlId = 0;

	CDXUTButton* button;
	hud.AddButton( nextControlId++, L"Toggle full screen", 5, 10, 125, 22, 0, false, &button );
	SasControl::Button* sasButton = new SasControl::Button(button, new FullScreenFunctor());
	idMap[button->GetID()] = sasButton;
	sasControls.push_back(sasButton);

	int nVariable = 0;
	ID3DX11EffectVariable* variable;
	while( (variable = effect->GetVariableByIndex(nVariable))->IsValid() )
	{
		ID3DX11EffectStringVariable* controlTypeString = variable->GetAnnotationByName("SasUiControl")->AsString();
		const char* controlTypeName = NULL;
		controlTypeString->GetString(&controlTypeName);
		if(controlTypeName)
		{
			if(strcmp(controlTypeName, "Slider") == 0)
			{
				SasControl::Slider* sasSlider = new SasControl::Slider(variable, ui, nextPosition, nextControlId, idMap);
				sasControls.push_back(sasSlider);
			}
			if(strcmp(controlTypeName, "CheckBox") == 0)
			{
				SasControl::Checkbox* sasCheckbox = new SasControl::Checkbox(variable, ui, nextPosition, nextControlId, idMap);
				sasControls.push_back(sasCheckbox);
			}
		}
		nVariable++;
	}

	ID3D11DeviceContext* immediateDeviceContext = DXUTGetD3D11DeviceContext();
	dialogResourceManager.OnD3D11CreateDevice( device, immediateDeviceContext );

	return S_OK;
}

HRESULT Dxa::Sas::createSwapChainResources()
{
	dialogResourceManager.OnD3D11ResizedSwapChain(device, &backbufferSurfaceDesc);

	hud.SetLocation( backbufferSurfaceDesc.Width-170, 0 );
    hud.SetSize( 170, 170 );
    ui.SetLocation( 0, 0 );
    ui.SetSize( 170, 500 );
	return S_OK;
}

HRESULT Dxa::Sas::releaseResources()
{
	effect->Release();
	dialogResourceManager.OnD3D11DestroyDevice();

	DXUTGetGlobalResourceCache().OnDestroyDevice();
	std::vector<SasControl::Base*>::iterator i = sasControls.begin();
	std::vector<SasControl::Base*>::iterator e = sasControls.end();
	while(i != e)
	{
		delete *i;
		i++;
	}

	return S_OK;
}

HRESULT Dxa::Sas::releaseSwapChainResources()
{
	dialogResourceManager.OnD3D11ReleasingSwapChain();

	return S_OK;
}

void Dxa::Sas::animate(double dt, double t)
{
	guiDt = dt;
}

bool Dxa::Sas::processMessage( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	if(uMsg == WM_KEYDOWN && wParam == VK_SPACE)
		hide = !hide;

	if(hide)
		return false;

	bool noFurther;
    noFurther = dialogResourceManager.MsgProc( hWnd, uMsg, wParam, lParam );
    if( noFurther )
        return true;

	// Give the dialogs a chance to handle the message first
    noFurther = hud.MsgProc( hWnd, uMsg, wParam, lParam );
    if( noFurther )
        return true;
    noFurther = ui.MsgProc( hWnd, uMsg, wParam, lParam );
    if( noFurther )
        return true;

	return false;
}

void Dxa::Sas::render(ID3D11DeviceContext* context)
{
	if(hide)
		return;
	#pragma region clear
	if(typeid(*this) == typeid(Dxa::Sas))
	{
		float clearColor[4] = { 0.0f, 0.0f, 0.5f, 0.0f };

		ID3D11RenderTargetView* defaultRtv = DXUTGetD3D11RenderTargetView();
		ID3D11DepthStencilView* defaultDsv = DXUTGetD3D11DepthStencilView();
		context->ClearRenderTargetView( defaultRtv, clearColor );
		context->ClearDepthStencilView( defaultDsv, D3D11_CLEAR_DEPTH, 1.0, 0 );
	}
	#pragma endregion a little cheat to clear the screen when testing Dxa::Sas without deriving from it

	if(swapChain == NULL)
		return;
	hud.OnRender( guiDt ); 
    ui.OnRender( guiDt );
}

void CALLBACK Dxa::Sas::onGuiEvent( UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext )
{
	((Dxa::Sas*)pUserContext)->processGuiEvent(nEvent, nControlID, pControl);
}

void Dxa::Sas::processGuiEvent( UINT nEvent, int nControlID, CDXUTControl* pControl)
{
	std::map<int, SasControl::Base*>::iterator iControl = idMap.find(nControlID);
	if(iControl != idMap.end())
		iControl->second->apply();
}

void Dxa::Sas::loadEffect()
{
	ID3D10Blob* compilationErrors = NULL;
	ID3D10Blob* compiledEffect = NULL;

	if(FAILED(
		D3DX11CompileFromFileW(
			L"fx/sasTest.fx", NULL, NULL, NULL,
			"fx_5_0", 0, 0, NULL, &compiledEffect, &compilationErrors, NULL))) {
				if(compilationErrors)
		MessageBoxA( NULL, 
			(LPSTR)compilationErrors->GetBufferPointer(),
			"Failed to load effect file!", MB_OK);
				else
			MessageBoxA( NULL, 
			"File cound not be opened",
			"Failed to load effect file!", MB_OK);
		exit(-1);
	 }

	if(FAILED(
	D3DX11CreateEffectFromMemory(
		compiledEffect->GetBufferPointer(), 
		compiledEffect->GetBufferSize(),
		0,
		device,
		&effect)  )) {
		MessageBoxA( NULL, 
			"D3DX11CreateEffectFromMemory failed.",
			"Failed to load effect file!", MB_OK);
		exit(-1);
	 }
}

CDXUTButton* Dxa::Sas::addButton(LPCWSTR strText, int x, int y, int width, int height, UINT nHotkey,
			bool bIsDefault, SasControl::Button::Functor* functor )
{
	CDXUTButton* button;
	hud.AddButton( nextControlId++, strText, x, y, width, height, nHotkey, false, &button );
	SasControl::Button* sasButton = new SasControl::Button(button, functor);
	idMap[button->GetID()] = sasButton;
	sasControls.push_back(sasButton);
	return button;
}