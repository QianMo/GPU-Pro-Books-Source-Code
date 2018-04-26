#pragma once
#include "Directory.h"

class XMLNode;
class ResourceSet;
//class CustomResourceDesc;

typedef std::map<const std::wstring, D3D10_BUFFER_DESC>			BufferDescDirectory;
typedef std::map<const std::wstring, D3D10_TEXTURE1D_DESC>			Texture1DDescDirectory;
typedef std::map<const std::wstring, D3D10_TEXTURE2D_DESC>			Texture2DDescDirectory;
typedef std::map<const std::wstring, D3D10_TEXTURE3D_DESC>			Texture3DDescDirectory;
struct SRVParameters{ D3D10_SHADER_RESOURCE_VIEW_DESC desc; std::wstring resourceName;};
typedef std::map<const std::wstring, SRVParameters> SRVDescDirectory;
struct RTVParameters{ D3D10_RENDER_TARGET_VIEW_DESC desc; std::wstring resourceName;};
typedef std::map<const std::wstring, RTVParameters>	RTVDescDirectory;
struct DSVParameters{ D3D10_DEPTH_STENCIL_VIEW_DESC desc; std::wstring resourceName;};
typedef std::map<const std::wstring, DSVParameters>	DSVDescDirectory;

//typedef std::map<std::wstring, CustomResourceDesc*>			CustomResourceDescDirectory;

class ResourceBuilder
{
	BufferDescDirectory		bufferDescDirectory;
	Texture1DDescDirectory	texture1DDescDirectory;
	Texture2DDescDirectory	texture2DDescDirectory;
	Texture3DDescDirectory	texture3DDescDirectory;
	SRVDescDirectory		srvDescDirectory;
	RTVDescDirectory		rtvDescDirectory;
	DSVDescDirectory		dsvDescDirectory;

	bool swapChainBound;

public:
	ResourceBuilder(XMLNode& resourcesNode, bool swapChainBound);
	void loadBuffers(XMLNode& resourcesNode);
	void loadTexture1Ds(XMLNode& resourcesNode);
	void loadTexture2Ds(XMLNode& resourcesNode);
	void loadShaderResourceViews(XMLNode& resourcesNode);
	void loadRenderTargetViews(XMLNode& resourcesNode);

	void loadVariables(XMLNode& variablesNode);

	void instantiate(ResourceSet* resourceSet, ID3D10Device* device);
	void defineVariables(ResourceSet* resourceSet, ID3D10Device* device);
};