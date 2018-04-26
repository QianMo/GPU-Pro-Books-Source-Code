#pragma once

#include "Directory.h"
#include "DXUTCamera.h"

class EnvironmentLightingScene
{
	typedef ResourceMap<const std::wstring, ID3D10Effect*>	EffectDirectory;
	
	typedef ResourceMap<const std::wstring, ID3DX10Mesh*>     MeshDirectory;
	/// D3D device reference
	ID3D10Device* device;

	IDXGISwapChain* swapChain;

	/// D3D effect reference
	ID3D10EffectPool* effectPool;
	EffectDirectory effects;

	/// Opens and compiles the .fx file, creating the effect instance.
	void loadEffectPool(const std::wstring& fileName);
	void loadChildEffect(const std::wstring& fileName, const std::wstring& name);

	/// Viewport width. Not set by constructor.
	unsigned int viewportWidth;
	/// Viewport height. Not set by constructor.
	unsigned int viewportHeight;

	CFirstPersonCamera camera;

	ID3D10ShaderResourceView* envMapCubeSRV;

	MeshDirectory meshDirectory;

	ID3D10InputLayout* inputLayout;
	ID3D10InputLayout* shadowInputLayout;
	ID3D10InputLayout* quadInputLayout;

	D3DXVECTOR3 sceneCenter;
	float sceneRadius;

	struct DirectionalLight{
		D3DXVECTOR4 position;
		D3DXVECTOR4 direction;
		D3DXVECTOR4 radiance;
		D3DXMATRIX  lightViewProjMatrix;
	};

	ID3D10Texture2D* depthStencilTextureArray;
	ID3D10Texture2D* shadowMapTextureArray;
	ID3D10ShaderResourceView* shadowMapSRV;
	ID3D10DepthStencilView* shadowMapDSV;
	ID3D10RenderTargetView* shadowMapRTV;


	typedef std::vector<DirectionalLight> DirectionalLightList;
	DirectionalLightList directionalLightList;

	ID3D10Buffer* directionalLightBuffer;
	
	unsigned int shadowMapHeight;
	unsigned int shadowMapWidth;

	ID3D10Texture2D* sysEnvTexture;
	D3D10_TEXTURE2D_DESC tDesc;
	void sampleCubeFace(unsigned int arraySlice, D3DXVECTOR3 onScreenWorldPos, D3DXVECTOR3 pixIncrement, D3DXVECTOR3 rowIncrement, D3DXVECTOR3 facing);
	void sampleCubeFaceErrorDiffusion(unsigned int arraySlice, D3DXVECTOR3 onScreenWorldPos, D3DXVECTOR3 pixIncrement, D3DXVECTOR3 rowIncrement, D3DXVECTOR3 facing);
	void samplePhiTheta();
	void samplePhiThetaErrorDiffusion();

	ID3D10RenderTargetView* swapChainRenderTargetView;
	ID3D10DepthStencilView* swapChainDepthStencilView;
public:
	/// Constructor.
	EnvironmentLightingScene(ID3D10Device* device);

	inline ID3D10Device* getDevice(){return device;}
	inline ID3D10Effect* getEffect(const std::wstring& effectName){EffectDirectory::iterator iEffect =  effects.find(effectName); if(iEffect != effects.end()) return iEffect->second; return NULL;}
	inline ID3D10Effect* getEffect(){return effectPool->AsEffect();}
	ID3D10EffectTechnique* getTechnique(const std::wstring& effectName, const std::string& techniqueName);

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

	ID3D10Buffer* getDirectionalLightBuffer(){return directionalLightBuffer;}
	ID3D10ShaderResourceView* getShadowMapSRV(){return shadowMapSRV;}

	void renderShadowMaps();

};
