#pragma once
#include "HOQDemo10.h"

#ifndef MAX_R2VB_SIZE
#define MAX_R2VB_SIZE 8192
#endif


class HierarchicalItemBuffer {
public:
	HierarchicalItemBuffer();

	HRESULT Init( ID3D10Device* d3dDevice, size_t itemWidth, size_t itemHeight, size_t histoWidth, size_t histoHeight );
	void Destroy();
	
	inline ID3D10RenderTargetView*		GetItemBufferRenderTargetView() const;
	inline ID3D10ShaderResourceView*	GetHistogramShaderResourceView() const;
	inline void							GetHistogramDimensions( float& histoWidth, float& histoHeight );
	inline void							GetHistogramDimensions( UINT& histoWidth, UINT& histoHeight );

	HRESULT ScatterToHistogram( ID3D10Device* d3dDevice, BOOL genMips = FALSE );



private:
	ID3D10Texture2D*			scatterTexture;
	ID3D10RenderTargetView*		scatterRenderTarget;
	ID3D10ShaderResourceView*	scatterShaderResource;

	ID3D10Buffer*				pointBuffer;

	ID3D10InputLayout*			scatterLayout3;

	ID3D10Effect*				effect;

	ID3D10EffectPass*			scatterPass2;
	ID3D10EffectPass*			scatterPass3;

	ID3D10RenderTargetView*		itemInputRenderTarget;
	ID3D10ShaderResourceView*	itemInputResource;

	ID3D10EffectShaderResourceVariable* inputTexture;

	HRESULT CreateRenderTargetAndResource( ID3D10Device* d3dDevice, size_t width, size_t height, DXGI_FORMAT format,
		ID3D10RenderTargetView** rtvOut, ID3D10ShaderResourceView** srvOut, UINT miscFlags = 0 ) const;

	HRESULT CreatePointVertexBuffer( ID3D10Device* d3dDevice, ID3D10Buffer** vbOut, size_t width, size_t height, size_t blocking ) const;
	HRESULT CreateScatterLayout3( ID3D10Device* d3dDevice, ID3D10InputLayout** inputLayout, ID3D10EffectPass* effect ) const;

	HRESULT CreateEffect( ID3D10Device* pd3dDevice, ID3D10Effect** effectOut, UINT uiCompileFlags ) const;
	bool	BindEffectVariables( ID3D10Effect* d3dEffect );
	
	HRESULT	CreateInputLayouts( ID3D10Device* d3dDevice, 
		ID3D10InputLayout** vbLayout, ID3D10InputLayout** scatterLayoutOut,
		ID3D10EffectPass* vbCopyPass, ID3D10EffectPass* scatterPass ) const;


	size_t						bucketCount;
	size_t						bucketWidth;
	size_t						bucketHeight;

	size_t						itemCount;
	size_t						width;
	size_t						height;

	size_t						levelCount;

	static const DXGI_FORMAT	format;
	static const UINT			stride;


};


ID3D10RenderTargetView* HierarchicalItemBuffer::GetItemBufferRenderTargetView() const {
	return itemInputRenderTarget;
}

ID3D10ShaderResourceView* HierarchicalItemBuffer::GetHistogramShaderResourceView() const {
	return scatterShaderResource;
}

void HierarchicalItemBuffer::GetHistogramDimensions( float& histoWidth, float& histoHeight ) {
	histoWidth = (float)bucketWidth;
	histoHeight= (float)bucketHeight;
}

void HierarchicalItemBuffer::GetHistogramDimensions( UINT& histoWidth, UINT& histoHeight ) {
	histoWidth = (UINT)bucketWidth;
	histoHeight= (UINT)bucketHeight;
}