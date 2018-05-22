#include <stdafx.h>
#include <DEMO.h>
#include <DX11_INDEX_BUFFER.h>

void DX11_INDEX_BUFFER::Release()
{
	SAFE_RELEASE(indexBuffer); 
	indices.Erase();
}

bool DX11_INDEX_BUFFER::Create(bool dynamic,int maxIndexCount)
{
	this->dynamic = dynamic;
	this->maxIndexCount = maxIndexCount;
	indices.Resize(maxIndexCount);

	D3D11_BUFFER_DESC bd;
	ZeroMemory(&bd,sizeof(bd));
	bd.ByteWidth = sizeof(int)*maxIndexCount;
	if(dynamic)
	{
		bd.Usage = D3D11_USAGE_DYNAMIC;
	  bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	}
	else
	{
	  bd.Usage = D3D11_USAGE_DEFAULT;
    bd.CPUAccessFlags = 0;
	}
	bd.BindFlags = D3D11_BIND_INDEX_BUFFER;
	
	if(DEMO::renderer->GetDevice()->CreateBuffer(&bd,NULL,&indexBuffer)!=S_OK)
		return false;

	return true;
}

int DX11_INDEX_BUFFER::AddIndices(int numIndices,const int *indices)
{
	if((numIndices<1)||(!indices))
		return -1;
	int firstListIndex = this->indices.AddElements(numIndices,indices);
	assert(this->indices.GetSize()<=maxIndexCount);
	return firstListIndex;
}

bool DX11_INDEX_BUFFER::Update()
{
	if(indices.GetSize()>0)
	{
    if(dynamic)
		{
			D3D11_MAPPED_SUBRESOURCE mappedResource;
			if(DEMO::renderer->GetDeviceContext()->Map(indexBuffer,0,D3D11_MAP_WRITE_DISCARD,0,&mappedResource)!=S_OK)
				return false;
      memcpy(mappedResource.pData,indices,sizeof(int)*indices.GetSize());
			DEMO::renderer->GetDeviceContext()->Unmap(indexBuffer,0);
		}
		else
		{
      DEMO::renderer->GetDeviceContext()->UpdateSubresource(indexBuffer,0,NULL,indices,0,0);
		}
	}
	return true;
}

void DX11_INDEX_BUFFER::Bind() const      
{
	DEMO::renderer->GetDeviceContext()->IASetIndexBuffer(indexBuffer,DXGI_FORMAT_R32_UINT,0);
}
