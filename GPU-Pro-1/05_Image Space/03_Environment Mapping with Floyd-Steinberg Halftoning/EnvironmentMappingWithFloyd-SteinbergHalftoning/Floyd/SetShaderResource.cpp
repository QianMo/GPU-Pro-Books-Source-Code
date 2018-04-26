#include "DXUT.h"
#include "SetShaderResource.h"
#include "Theatre.h"
#include "ResourceLocator.h"
#include "ScriptShaderResourceViewVariable.h"

SetShaderResource::SetShaderResource(ID3D10EffectShaderResourceVariable* effectVariable, ScriptShaderResourceViewVariable* srv)
{
	this->srv = srv;
	this->effectVariable = effectVariable;
}

void SetShaderResource::execute(const TaskContext& context)
{
	ID3D10ShaderResourceView* asrv = srv->getShaderResourceView();
	if(srv)
		effectVariable->SetResource(asrv);
}
