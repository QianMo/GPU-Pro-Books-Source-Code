#include "Model.h"
#include "MemoryLeakTracker.h"

CoreResult Model::Init(Core *core, std::wstring &name, void *indexBufferData, DWORD indexCount, DXGI_FORMAT indexBufferFormat, void *vertexBufferData, UINT bufferElementSize, DWORD vertexCount, D3D11_INPUT_ELEMENT_DESC *inputElementDesc, UINT numInputElements, D3D11_PRIMITIVE_TOPOLOGY primitiveTopology, CoreColor &diffuseColor, CoreTexture2D *texture)
{
	D3D11_BUFFER_DESC bufferDesc;
	D3D11_SUBRESOURCE_DATA InitData;
	HRESULT hr;

	this->name = name;
	this->core = core;
	this->indexCount = indexCount;
	this->bufferElementSize = bufferElementSize;
	this->primitiveTopology = primitiveTopology;
	this->vertexCount = vertexCount;
	this->indexBufferFormat = indexBufferFormat;
	this->inputElementDesc = new D3D11_INPUT_ELEMENT_DESC[numInputElements];
	memcpy(this->inputElementDesc, inputElementDesc, sizeof(D3D11_INPUT_ELEMENT_DESC) * numInputElements);
	this->numInputElements = numInputElements;
	this->diffuseColor = diffuseColor;
	this->texture = texture;

	// Create D3D11 Vertex Buffer
	
	bufferDesc.Usage            = D3D11_USAGE_DEFAULT;
	bufferDesc.ByteWidth        = bufferElementSize * vertexCount;
	bufferDesc.BindFlags        = D3D11_BIND_VERTEX_BUFFER;
	bufferDesc.CPUAccessFlags   = 0;
	bufferDesc.MiscFlags        = 0;

	
	InitData.pSysMem = vertexBufferData;
	InitData.SysMemPitch = 0;
	InitData.SysMemSlicePitch = 0;
	hr = core->GetDevice()->CreateBuffer(&bufferDesc, &InitData, &vertexBuffer);

	if(FAILED(hr))
	{
		CoreLog::Information(L"Error creating vertex buffer, HRESULT = %x", hr);
		return CORE_MISC_ERROR;
	}
	
	if(indexBufferData)
	{
		// Create D3D11 Index Buffer
		bufferDesc.Usage           = D3D11_USAGE_DEFAULT;
		bufferDesc.ByteWidth       = sizeof(unsigned int) * indexCount;
		bufferDesc.BindFlags       = D3D11_BIND_INDEX_BUFFER;
		bufferDesc.CPUAccessFlags  = 0;
		bufferDesc.MiscFlags       = 0;

		InitData.pSysMem = (void *)indexBufferData;
		InitData.SysMemPitch = 0;
		InitData.SysMemSlicePitch = 0;
		hr = core->GetDevice()->CreateBuffer(&bufferDesc, &InitData, &indexBuffer);   
		if(FAILED(hr))
		{
			SAFE_RELEASE(vertexBuffer);
			CoreLog::Information(L"Error creating index buffer, HRESULT = %x", hr);
			return CORE_MISC_ERROR;
		}
	}
	else
		indexBuffer = NULL;

	if(texture)
	{
		D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;

		srvDesc.Texture2D.MipLevels = 1;
		srvDesc.Texture2D.MostDetailedMip = 0;
		srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
		srvDesc.Format = texture->GetFormat();
		
		CoreResult cr = texture->CreateShaderResourceView(&srvDesc, &textureSRV);
		if(cr != CORE_OK)
			return cr;
	}
	else
		textureSRV = NULL;

	return CORE_OK;
}

Model::Model()
{
	vertexBuffer = NULL;
	indexBuffer = NULL;
	inputElementDesc = NULL;
}

void Model::finalRelease()
{
	SAFE_RELEASE(vertexBuffer);
	SAFE_RELEASE(indexBuffer);
	SAFE_DELETE(inputElementDesc);
	SAFE_RELEASE(textureSRV);
	SAFE_RELEASE(texture);
}

CoreResult Model::CreateInputLayout(ID3DX11EffectPass *pass, ID3D11InputLayout** outLayout)
{
	D3DX11_PASS_DESC passDesc;
    pass->GetDesc(&passDesc);
	HRESULT result = core->GetDevice()->CreateInputLayout(inputElementDesc, numInputElements, passDesc.pIAInputSignature, passDesc.IAInputSignatureSize, outLayout);

	if(FAILED(result))
	{
		CoreLog::Information(L"Error creating the InputLayout, HRESULT = %x", result);
		return CORE_MISC_ERROR;
	}
	return CORE_OK;
}

void Model::SetMaterial(ID3DX11EffectVectorVariable *diffuseColorVariable, ID3DX11EffectShaderResourceVariable *textureVariable)
{
	if(diffuseColorVariable) diffuseColorVariable->SetFloatVector(diffuseColor.arr);
	if(textureVariable)	textureVariable->SetResource(textureSRV);
}

void Model::Draw()
{
	UINT strides = bufferElementSize;
    UINT offsets = 0;
	
	core->GetImmediateDeviceContext()->IASetVertexBuffers(0, 1, &vertexBuffer, &strides, &offsets);
	core->GetImmediateDeviceContext()->IASetPrimitiveTopology(primitiveTopology);

	if(indexBuffer)
	{
		core->GetImmediateDeviceContext()->IASetIndexBuffer(indexBuffer, indexBufferFormat, 0);
		core->GetImmediateDeviceContext()->DrawIndexed(indexCount, 0, 0);
	}
	else
		core->GetImmediateDeviceContext()->Draw(vertexCount, 0);
}