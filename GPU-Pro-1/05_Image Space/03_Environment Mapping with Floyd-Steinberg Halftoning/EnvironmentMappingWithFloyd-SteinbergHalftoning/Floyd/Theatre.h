#pragma once
#include "Directory.h"
#include "ControlContext.h"
#include "SDKMisc.h"

#include "Play.h"
#include "ControlStatus.h"

class XMLNode;
class NodeGroup;
class Scene;
class PropsMaster;
class FileFinder;
class ResourceLocator;

class Theatre
{
	/// D3D device reference
	ID3D10Device* device;

	IDXGISwapChain* swapChain;

	/// D3D effect reference
	ID3D10EffectPool* effectPool;
	EffectDirectory effects;

	ResourceLocator* resourceLocator;

	/// Input status.
	ControlStatus controlStatus;

	Play* play;
	bool loaded;

	void loadPlay(XMLNode& playNode);

	/// Opens and compiles the .fx file, creating the effect instance.
	void loadEffectPool(const std::wstring& fileName);

	void loadChildEffect(const std::wstring& fileName, const std::wstring& name);

public:
	/// Constructor.
	Theatre(ID3D10Device* device);

	inline ID3D10Device* getDevice(){return device;}
	inline ID3D10Effect* getEffect(const std::wstring& effectName){EffectDirectory::iterator iEffect =  effects.find(effectName); if(iEffect != effects.end()) return iEffect->second; return NULL;}
	inline ID3D10Effect* getEffect(){return effectPool->AsEffect();}
	ID3D10EffectTechnique* getTechnique(const std::wstring& effectName, const std::string& techniqueName);

	inline PropsMaster* getPropsMaster(){return play->getPropsMaster();}
	inline FileFinder* getFileFinder(){return play->getFileFinder();}
	inline Play* getPlay(){return play;}
	ResourceLocator* getResourceLocator(){return resourceLocator;}

	void setSwapChain(IDXGISwapChain* swapChain){this->swapChain = swapChain;}

	/// Creates D3D resources.
	HRESULT createResources();
	/// Releases D3D resources.
	HRESULT releaseResources();

	/// Creates swap-chain dependent D3D resources.
	HRESULT createSwapChainResources();
	/// Releases swap-chain dependent D3D resources.
	HRESULT releaseSwapChainResources();

	/// Handles user input.
	void processMessage( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* trapped);
	/// Animates cameras and theatre objects.
	void animate(double dt, double t);

	/// Renders the scene.
	void render();

	static const unsigned int actActivate;
};
