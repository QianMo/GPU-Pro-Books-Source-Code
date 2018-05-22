#include <stdafx.h>
#include <DEMO.h>
#include <DX11_UNIFORM_BUFFER.h>

void DX11_UNIFORM_BUFFER::Release()
{
  SAFE_RELEASE(uniformBuffer);
}

bool DX11_UNIFORM_BUFFER::Create(uniformBufferBP bindingPoint,const UNIFORM_LIST &uniformList)
{
	this->bindingPoint = bindingPoint;
	
	for(int i=0;i<uniformList.GetSize();i++)
	{
		UNIFORM* uniform = uniformList.GetElement(i);
		switch(uniform->dataType)
		{
		case INT_DT:
		case FLOAT_DT:
      size += uniform->count;
			break;
		case VEC2_DT:
			size += uniform->count*2;
			break;
		case VEC3_DT:
			size += uniform->count*3;
			break;
		case VEC4_DT:
			size += uniform->count*4;
			break;
		case MAT4_DT:
			size += uniform->count*16;
			break;
		}
	}

	D3D11_BUFFER_DESC bd;
	ZeroMemory(&bd,sizeof(bd));
	bd.Usage = D3D11_USAGE_DYNAMIC;
	bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	int bufferSize = size;
	int align = bufferSize % 4;
	if(align>0)
		bufferSize += 4-align;
	bd.ByteWidth = bufferSize*sizeof(float);
	bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	if(DEMO::renderer->GetDevice()->CreateBuffer(&bd,NULL,&uniformBuffer)!=S_OK)
		return false;

	return true;
}

bool DX11_UNIFORM_BUFFER::Update(float *uniformBufferData) 
{
	if(!uniformBufferData)
		return false;
	
	D3D11_MAPPED_SUBRESOURCE MappedResource;
	DEMO::renderer->GetDeviceContext()->Map(uniformBuffer,0,D3D11_MAP_WRITE_DISCARD,0,&MappedResource);
	float *resourceData = (float*)MappedResource.pData;
  memcpy(resourceData,uniformBufferData,size*sizeof(float));
  DEMO::renderer->GetDeviceContext()->Unmap(uniformBuffer,0);

	return true;
}

void DX11_UNIFORM_BUFFER::Bind(shaderTypes shaderType) const
{
	switch(shaderType)
	{
	case VERTEX_SHADER:
		DEMO::renderer->GetDeviceContext()->VSSetConstantBuffers(bindingPoint,1,&uniformBuffer);
		break;

	case GEOMETRY_SHADER:
		DEMO::renderer->GetDeviceContext()->GSSetConstantBuffers(bindingPoint,1,&uniformBuffer);
		break;

	case FRAGMENT_SHADER: 
		DEMO::renderer->GetDeviceContext()->PSSetConstantBuffers(bindingPoint,1,&uniformBuffer);
		break;

	case COMPUTE_SHADER: 
		DEMO::renderer->GetDeviceContext()->CSSetConstantBuffers(bindingPoint,1,&uniformBuffer);
		break;
	}
}

