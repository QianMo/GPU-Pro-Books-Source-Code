#pragma once

#include <istream>

#include "CoreError.h"
#include "CoreColor.h"
#include "ICoreBase.h"
#include "CoreLog.h"

class Core;

class CoreTexture3D : public ICoreBase
{
friend Core;
protected:
	// Init constructor
	CoreTexture3D();

	ID3D11Texture3D* texture;
	ID3D11Texture3D* gradient;
	Core* core;

	UINT width;
	UINT height;
	UINT depth;
	UINT mipLevels;
	UINT cpuAccessFlags;
	UINT miscFlags;
	D3D11_USAGE usage;
	UINT bindFlags;
	DXGI_FORMAT format;
	DWORD *histogram;
	DWORD histSize;
	DWORD maxHistValue;
	UINT maxVal;

	// Load a texture from a stream
	CoreResult init(Core* core, std::istream& in, UINT mipLevels,
			  	    UINT cpuAccessFlags, UINT miscFlags, D3D11_USAGE usage, UINT bindFlags);

	// Load a 3D texture from memory PARTIALLY BROKEN
	CoreResult init(Core* core, BYTE* data, UINT width, UINT height, UINT depth, UINT mipLevels,
		  	    DXGI_FORMAT format, UINT cpuAccessFlags, UINT miscFlags, D3D11_USAGE usage, UINT bindFlags);


	// Creates and fills the texture with the supplied data
	CoreResult createAndFillTexture(BYTE* data);
	// Loads a DAT from a stream
	CoreResult loadDat(std::istream& in, BYTE** data);

	// Creates and fills the gradient texture with the supplied data
	CoreResult createAndFillGradient(BYTE* data);

private:
	void calculateGradient(BYTE* data, BYTE **grad);

	//Helper function for gradient calculation
	void fillzSlice (float *zSlice, BYTE *data, int prevX, int x, int nextX, int prevY, int y, int nextY, int z);
	void calculateGradientVector(float * samplesNegZOffset, float * samplesNoZOffset, float * samplesPosZOffset, CoreVector3 &gradVector);
	
	// CleanUp
	virtual void finalRelease();
public:
	// Create a RenderTargetView
		CoreResult CreateRenderTargetView(D3D11_RENDER_TARGET_VIEW_DESC* rtvDesc, ID3D11RenderTargetView** rtv);
		// Create a DepthStencilView 
		CoreResult CreateDepthStencilView(D3D11_DEPTH_STENCIL_VIEW_DESC* dsvDesc, ID3D11DepthStencilView** dsv);
		// Create a ShaderResourceView
		CoreResult CreateShaderResourceView(D3D11_SHADER_RESOURCE_VIEW_DESC* srvDesc, ID3D11ShaderResourceView** srv);

		// Create a ShaderResourceView for the gradient
		CoreResult CreateShaderResourceViewGradient(D3D11_SHADER_RESOURCE_VIEW_DESC* srvDesc, ID3D11ShaderResourceView** srv);

		// Calculates the histogram for the 3D Texture
		void GetHistogram(DWORD **outHist, DWORD *outArrSize, DWORD *outMaxHistValue);


		inline DXGI_FORMAT GetFormat()						{ return format; }

		inline UINT GetWidth()								{ return width; }
		inline UINT GetHeight()								{ return height; }
		inline UINT GetDepth()								{ return depth; }
		inline ID3D11Resource* GetResource()				{ return texture; }
};