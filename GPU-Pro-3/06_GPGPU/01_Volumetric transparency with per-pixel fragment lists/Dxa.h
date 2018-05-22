#pragma once

namespace Dxa {

	/// Basic application class.
	class Base
	{
		#pragma region 0.0

	protected:
		ID3D11Device* device;
		IDXGISwapChain* swapChain;
		DXGI_SURFACE_DESC backbufferSurfaceDesc;
	public:
		Base(ID3D11Device* device)
			{this->device = device; swapChain = NULL;}
		void setSwapChain(IDXGISwapChain* swapChain,
		 const DXGI_SURFACE_DESC* backbufferSurfaceDesc) {
			 this->swapChain = swapChain;
			 this->backbufferSurfaceDesc =
				 *backbufferSurfaceDesc;}
		virtual ~Base(){}
		#pragma endregion members and initializers

		#pragma region 0.1
		virtual HRESULT createResources()
			{return S_OK;}
		virtual HRESULT createSwapChainResources()
			{return S_OK;}
		virtual HRESULT releaseResources()
			{return S_OK;}
		virtual HRESULT releaseSwapChainResources()
			{return S_OK;}
		virtual void animate(double dt, double t){}
		virtual bool processMessage( HWND hWnd, 
			UINT uMsg, WPARAM wParam, LPARAM lParam)
			{return false;}
		virtual void render(
			ID3D11DeviceContext* context){}
		#pragma endregion event handler methods
	};
}