#pragma once


#include "types.h"
#include <essentials/main.h>

#include <d3d11_1.h>
#include <d3dcompiler.h>


using namespace NEssentials;


namespace NGPU
{
	extern D3D_DRIVER_TYPE driverType;
	extern D3D_FEATURE_LEVEL featureLevel;
	extern ID3D11Device* device;
	extern ID3D11Device1* device1;
	extern ID3D11DeviceContext* deviceContext;
	extern ID3D11DeviceContext1* deviceContext1;
	extern IDXGISwapChain* swapChain;
	extern IDXGISwapChain1* swapChain1;
	extern ID3D11RenderTargetView* backBufferRTV;

	//

	void CreateD3D11(int width, int height);
	void DestroyD3D11();

	bool CompileShaderFromFile(const string& path, const string& entryPointName, const string& shaderModelName, const string& shaderMacros, ID3DBlob*& blob);

	void CreateRenderTarget(int width, int height, DXGI_FORMAT format, RenderTarget& renderTarget);
	void CreateDepthStencilTarget(int width, int height, DepthStencilTarget& depthStencilTarget);
	void CreateTexture(int width, int height, Texture& texture);
	bool CreateVertexShader(const string& path, const string& shaderMacros, ID3D11VertexShader*& vertexShader);
	bool CreatePixelShader(const string& path, const string& shaderMacros, ID3D11PixelShader*& pixelShader);
	bool CreateVertexShader(const string& path, ID3D11VertexShader*& vertexShader);
	bool CreatePixelShader(const string& path, ID3D11PixelShader*& pixelShader);
	bool CreateInputLayout(const string& dummyVertexShaderPath, D3D11_INPUT_ELEMENT_DESC inputLayoutElements[], int inputLayoutElementsCount, ID3D11InputLayout*& inputLayout);
	void CreateVertexBuffer(uint8* data, int dataSize, ID3D11Buffer*& vertexBuffer);
	void CreateIndexBuffer(uint8* data, int dataSize, ID3D11Buffer*& indexBuffer);
	void CreateConstantBuffer(int dataSize, ID3D11Buffer*& constantBuffer);
	void CreateSamplerState(ID3D11SamplerState*& samplerState, SamplerFilter filter, SamplerAddressing addressing, SamplerComparisonFunction comparisonFunction);
	void CreateRasterizerState(ID3D11RasterizerState*& rasterizerState);

	void DestroyRenderTarget(RenderTarget& renderTarget);
	void DestroyDepthStencilTarget(DepthStencilTarget& depthStencilTarget);
	void DestroyTexture(Texture& texture);
	void DestroyVertexShader(ID3D11VertexShader*& vertexShader);
	void DestroyPixelShader(ID3D11PixelShader*& pixelShader);
	void DestroyInputLayout(ID3D11InputLayout*& inputLayout);
	void DestroyBuffer(ID3D11Buffer*& buffer);
	void DestroySamplerState(ID3D11SamplerState*& samplerState);
	void DestroyRasterizerState(ID3D11RasterizerState*& rasterizerState);

	void UpdateTexture(Texture& texture, int mipmapIndex, uint8* data, int rowPitch);

	void SetViewport(int width, int height);
}
