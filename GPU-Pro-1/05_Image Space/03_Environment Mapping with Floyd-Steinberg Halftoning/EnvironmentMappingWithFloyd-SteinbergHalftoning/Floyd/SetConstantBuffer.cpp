#include "DXUT.h"
#include "SetConstantBuffer.h"
#include "ScriptResourceVariable.h"

SetConstantBuffer::SetConstantBuffer(ID3D10EffectConstantBuffer* effectConstantBuffer, ScriptResourceVariable* resource)
{
	this->effectConstantBuffer = effectConstantBuffer;
	this->resource = resource;
}

void SetConstantBuffer::execute(const TaskContext& context)
{
	ID3D10Resource* aresource = resource->getResource();
	if(aresource)
		effectConstantBuffer->SetConstantBuffer((ID3D10Buffer*)aresource);
}