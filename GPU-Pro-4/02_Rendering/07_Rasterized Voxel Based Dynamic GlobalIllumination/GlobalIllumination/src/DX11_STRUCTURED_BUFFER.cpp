#include <stdafx.h>
#include <DEMO.h>
#include <DX11_STRUCTURED_BUFFER.h>

void DX11_STRUCTURED_BUFFER::Release()
{
	SAFE_RELEASE(structuredBuffer);
	SAFE_RELEASE(unorderedAccessView);
  SAFE_RELEASE(shaderResourceView);
}

bool DX11_STRUCTURED_BUFFER::Create(int bindingPoint,int elementCount,int elementSize)
{
	this->bindingPoint = bindingPoint;
	this->elementCount = elementCount;
	this->elementSize = elementSize;

	D3D11_BUFFER_DESC bufferDesc;
	int stride = elementSize;
	bufferDesc.ByteWidth = stride*elementCount;
	bufferDesc.Usage = D3D11_USAGE_DEFAULT;
	bufferDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
	bufferDesc.CPUAccessFlags = 0;
	bufferDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
	bufferDesc.StructureByteStride = stride;
	if(DEMO::renderer->GetDevice()->CreateBuffer(&bufferDesc,NULL,&structuredBuffer)!=S_OK)
		return false;

	D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc;
	uavDesc.Format = DXGI_FORMAT_UNKNOWN;
	uavDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
	uavDesc.Buffer.FirstElement = 0;
	uavDesc.Buffer.Flags = 0;
	uavDesc.Buffer.NumElements = elementCount; 
	if(DEMO::renderer->GetDevice()->CreateUnorderedAccessView(structuredBuffer,&uavDesc,&unorderedAccessView)!=S_OK)
		return false;
	
	D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
	srvDesc.Format = DXGI_FORMAT_UNKNOWN;
	srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
	srvDesc.Buffer.ElementOffset = 0;
	srvDesc.Buffer.ElementWidth = elementCount;
	if(DEMO::renderer->GetDevice()->CreateShaderResourceView(structuredBuffer,&srvDesc,&shaderResourceView)!=S_OK) 
		return false;

	return true;
}

void DX11_STRUCTURED_BUFFER::Bind(shaderTypes shaderType) const
{
	switch(shaderType)
	{
	case VERTEX_SHADER:
		DEMO::renderer->GetDeviceContext()->VSSetShaderResources(bindingPoint,1,&shaderResourceView);
		break;

	case GEOMETRY_SHADER:
		DEMO::renderer->GetDeviceContext()->GSSetShaderResources(bindingPoint,1,&shaderResourceView);
		break;

	case FRAGMENT_SHADER: 
		DEMO::renderer->GetDeviceContext()->PSSetShaderResources(bindingPoint,1,&shaderResourceView);
		break;

	case COMPUTE_SHADER:
		DEMO::renderer->GetDeviceContext()->CSSetShaderResources(bindingPoint,1,&shaderResourceView);
		break;
	}
}

ID3D11UnorderedAccessView* DX11_STRUCTURED_BUFFER::GetUnorderdAccessView() const
{
	return unorderedAccessView; 
}


