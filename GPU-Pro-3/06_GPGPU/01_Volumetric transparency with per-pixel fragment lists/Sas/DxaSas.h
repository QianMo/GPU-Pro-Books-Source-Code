#pragma once
#include "../Dxa.h"
#include <map>
#include <vector>
#include "DXUTgui.h"
#include "d3dx11effect.h"
#include "SasButton.h"

namespace SasControl
{
	class Base;
}

namespace Dxa {

	/// Basic application class extension displaying DXUT GUI elements to control Standard Annotation Syntax effect variables.
	class Sas :
		public Dxa::Base
	{
	private:
		bool hide;
		float guiDt;
		int nextControlId;
	protected:
		CDXUTDialog hud; // manages the 3D UI
		CDXUTDialog ui; // dialog for sample specific controls
		CDXUTDialogResourceManager dialogResourceManager; // manager for shared resources of dialogs

		static void CALLBACK onGuiEvent( UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext );
		void processGuiEvent( UINT nEvent, int nControlID, CDXUTControl* pControl);

		virtual void loadEffect();
		ID3DX11Effect* effect;

		CDXUTButton* addButton(LPCWSTR strText, int x, int y, int width, int height, UINT nHotkey,
			bool bIsDefault, SasControl::Button::Functor* functor );

		std::map<int, SasControl::Base*> idMap;
		std::vector<SasControl::Base*> sasControls;

	public:
		Sas(ID3D11Device* device);
		virtual ~Sas();
		virtual HRESULT createResources();
		virtual HRESULT createSwapChainResources();
		virtual HRESULT releaseResources();
		virtual HRESULT releaseSwapChainResources();

		virtual void animate(double dt, double t);
		virtual bool processMessage( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
		virtual void render(ID3D11DeviceContext* context);

	};

}