/*
**********************************************************************
 * Demo program for
 * Rule-based Geometry Synthesis in Real-time
 * ShaderX 8 article.
 *
 * @author: Laszlo Szecsi, used with permission
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted for any non-commercial programs.
 * 
 * Use it for your own risk. The author(s) do(es) not take
 * responsibility or liability for the damages or harms caused by
 * this software.
**********************************************************************
*/

#pragma once

/////////////////////////////////////////////////////////////////////////
// EngineInterface class - an interface for DirectX 10 rendering engines
/////////////////////////////////////////////////////////////////////////

class EngineInterface
{
protected:
	ID3D10Device* device;
	IDXGISwapChain* swapChain;
public:
	EngineInterface(ID3D10Device* device){this->device = device;}
	void setSwapChain(IDXGISwapChain* swapChain){this->swapChain = swapChain;}
	virtual ~EngineInterface(){}

  virtual HRESULT createResources() = 0;
	virtual HRESULT createSwapChainResources() = 0;
	virtual HRESULT releaseResources() = 0;
	virtual HRESULT releaseSwapChainResources() = 0;

	virtual void animate(double dt, double t) = 0;
	virtual void processMessage( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) = 0;
	virtual void handleKeyboard( UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext ) = 0;
	virtual void render() = 0;

};
