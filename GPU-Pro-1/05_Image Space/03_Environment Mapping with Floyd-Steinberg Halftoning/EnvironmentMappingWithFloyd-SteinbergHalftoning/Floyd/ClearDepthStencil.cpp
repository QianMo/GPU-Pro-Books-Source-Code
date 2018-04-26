#include "DXUT.h"
#include "ClearDepthStencil.h"
#include "Theatre.h"
#include "ResourceOwner.h"
#include "ResourceLocator.h"
#include "ScriptDepthStencilViewVariable.h"
#include "TaskContext.h"

ClearDepthStencil::ClearDepthStencil(ScriptDepthStencilViewVariable* dsv, unsigned int flags, float depth, unsigned char stencil)
{
	this->dsv = dsv;
	this->depth = depth;
	this->stencil = stencil;
	this->flags = flags;
}

void ClearDepthStencil::execute(const TaskContext& context)
{
	ID3D10DepthStencilView* adsv = dsv->getDepthStencilView();
	if(adsv)
		context.theatre->getDevice()->ClearDepthStencilView(adsv, flags, depth, stencil);
}