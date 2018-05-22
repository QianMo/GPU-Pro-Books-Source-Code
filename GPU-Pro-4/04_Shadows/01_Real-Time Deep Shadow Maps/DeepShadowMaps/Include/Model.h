#pragma once
#include "Core.h"


class Model : public ICoreBase
{
public:
	Model();
	
	CoreResult Init(Core *core, std::wstring &name, void *indexBufferData, DWORD indexCount, DXGI_FORMAT indexBufferFormat, void *vertexBufferData, UINT bufferElementSize, DWORD vertexCount, D3D11_INPUT_ELEMENT_DESC *inputElementDesc, UINT numInputElements, D3D11_PRIMITIVE_TOPOLOGY primitiveTopology, CoreColor &diffuseColor, CoreTexture2D *texture);

	CoreResult CreateInputLayout(ID3DX11EffectPass *pass, ID3D11InputLayout** outLayout);

	void SetMaterial(ID3DX11EffectVectorVariable *diffuseColorVariable, ID3DX11EffectShaderResourceVariable *textureVariable);

	void Draw();

	inline bool HasIndexBuffer() { return indexBuffer != NULL; }; 
	inline CoreColor GetDiffuseColor() { return diffuseColor; };
	inline void SetDiffuseColor(CoreColor &diffuseColor) { this->diffuseColor = diffuseColor; };

protected:
	// CleanUp
	virtual void finalRelease();

	std::wstring				name;
	DWORD						indexCount;
	DXGI_FORMAT					indexBufferFormat;
	UINT						bufferElementSize;
	DWORD						vertexCount;
	D3D11_INPUT_ELEMENT_DESC	*inputElementDesc;
	UINT						numInputElements;
	ID3D11Buffer				*indexBuffer;
	ID3D11Buffer				*vertexBuffer;
	Core						*core;
	D3D11_PRIMITIVE_TOPOLOGY	primitiveTopology;
	CoreColor					diffuseColor;
	CoreTexture2D				*texture;
	ID3D11ShaderResourceView	*textureSRV;
};