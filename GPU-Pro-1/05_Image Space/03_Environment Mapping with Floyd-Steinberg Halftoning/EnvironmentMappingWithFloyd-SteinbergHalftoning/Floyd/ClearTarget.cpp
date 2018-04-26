#include "DXUT.h"
#include "ClearTarget.h"
#include "Theatre.h"
#include "ResourceLocator.h"
#include "ScriptRenderTargetViewVariable.h"
#include "TaskContext.h"

ClearTarget::ClearTarget(ScriptRenderTargetViewVariable* rtv, const D3DXVECTOR4& color)
{
	this->rtv = rtv;
	this->color = color;
}

void ClearTarget::execute(const TaskContext& context)
{
	ID3D10RenderTargetView* artv = rtv->getRenderTargetView();
	if(artv)
		context.theatre->getDevice()->ClearRenderTargetView(artv, color);
}