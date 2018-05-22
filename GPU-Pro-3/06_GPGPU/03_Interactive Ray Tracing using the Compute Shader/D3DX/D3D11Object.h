// ================================================================================ //
// Copyright (c) 2011 Arturo Garcia, Francisco Avila, Sergio Murguia and Leo Reyes	//
//																					//
// Permission is hereby granted, free of charge, to any person obtaining a copy of	//
// this software and associated documentation files (the "Software"), to deal in	//
// the Software without restriction, including without limitation the rights to		//
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies	//
// of the Software, and to permit persons to whom the Software is furnished to do	//
// so, subject to the following conditions:											//
//																					//
// The above copyright notice and this permission notice shall be included in all	//
// copies or substantial portions of the Software.									//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR		//
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,			//
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE		//
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER			//
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,	//
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE	//
// SOFTWARE.																		//
// ================================================================================ //

// ------------------------------------------------
// D3D11Object.h
// ------------------------------------------------
// Helper class for DirectX11.0.
// Call this dogma of faith.

#ifndef D3DUTIL_H
#define D3DUTIL_H

#include "Common.h"
#ifdef WINDOWS
#include "WindowsUtil.h"

#include <D3DX11tex.h>
#include <D3DX11.h>
#include <D3DX11core.h>
#include <D3DX11async.h>
#include <D3Dcompiler.h>

#include <iostream>
#include <sstream>

template <class T>
class ResourceContainer
{
public:
	void AddComponent(ID3D11Resource* pResource)
	{
		m_vContainer.push_back( pResource );
	}

	void GetContainer() { return m_vContainer; }
private:
	vector<T>	m_vContainer;
};

class D3D11Object
{
private:
	// Basic variables for the D3DX-based appliaction
	ID3D11Device*				m_pDevice;
	IDXGISwapChain*				m_pSwapChain;
	ID3D11DeviceContext*		m_pDeviceContext;
	ID3D11RenderTargetView*		m_pRenderTargetView;
	ID3D11Texture2D*			m_pBackBufferTexture;
	ID3D11SamplerState*			m_pSamplerLinear;
	char						m_sVideoCardDescription[256];

public:
	D3D11Object( void );
	~D3D11Object( void );

	HRESULT						Initialize( HWND&, ID3D11UnorderedAccessView** );
	void						Shutdown();
	
	ID3D11Device*				GetDevice() { return m_pDevice; }
	IDXGISwapChain*				GetSwapChain() { return m_pSwapChain; };
	ID3D11DeviceContext*		GetDeviceContext() { return m_pDeviceContext; }
	ID3D11RenderTargetView*		GetRenderTargetView() { return m_pRenderTargetView; }
	ID3D11Texture2D*			GetBackBufferTexture() { return m_pBackBufferTexture; }
	char*						GetVideoCardDescription() { return m_sVideoCardDescription; }

	void						SetDevice( ID3D11Device* );
	void						SetSwapChain( IDXGISwapChain* );
	void						SetDeviceContext( ID3D11DeviceContext* );
	void						SetRenderTargetView( ID3D11RenderTargetView* );
	void						SetBackBufferTexture( ID3D11Texture2D* );
	void						SetVideoCardDescription( char* );

	// Heper functions
	HRESULT						CreateSamplerStates();
	HRESULT						TakeScreenshot();
	HRESULT						CompileShaderFromFile( WCHAR* szFileName, LPCSTR szEntryPoint, LPCSTR szShaderModel, ID3DBlob** ppBlobOut );

	HRESULT						FindDXSDKShaderFileCch( __in_ecount(cchDest) WCHAR* strDestPath,
									int cchDest, 
									__in LPCWSTR strFilename );
	HRESULT						CreateBufferTexUAV( ID3D11Texture2D* pTexture, ID3D11UnorderedAccessView** ppUAVOut );
	void						CreateViewport( UINT width, UINT height );

	ID3D11Buffer*				CreateAndCopyToDebugBuf( ID3D11Buffer* pBuffer );
	HRESULT						CreateRTVandUAV( ID3D11UnorderedAccessView* pUAV, ID3D11Texture2D* pBackBufferTexture = NULL );
	void						UpdateTime(float& time);
	void						InitDeviceContext( int HEIGHT, int WIDTH,HWND& hWnd );

	HRESULT						LoadSamplerState(ID3D11SamplerState** pSamplerLinear);

	HRESULT						CreateEnvironmentMap( LPCSTR a_FileName, ID3D11ShaderResourceView** a_pSRV );
	HRESULT						CreateTextureInArray( LPCSTR a_FileName, D3DX11_IMAGE_LOAD_INFO* a_LoadInfo, ID3D11Resource** a_pResource, ID3D11Resource* a_pDest, int a_iArraySlice );
	HRESULT						CreateRandomTexture( ID3D11Texture2D* a_pTexture2D, UINT a_iWidth, UINT a_iHeight );
};

#endif
#endif 