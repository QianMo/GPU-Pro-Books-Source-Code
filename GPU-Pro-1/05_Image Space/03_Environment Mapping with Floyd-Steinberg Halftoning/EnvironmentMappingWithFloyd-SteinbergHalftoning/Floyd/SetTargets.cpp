#include "DXUT.h"
#include "SetTargets.h"
#include "ResourceLocator.h"
#include "Theatre.h"
#include "ScriptDepthStencilViewVariable.h"
#include "ScriptRenderTargetViewVariable.h"
#include "TaskContext.h"

SetTargets::SetTargets(ScriptDepthStencilViewVariable* dsv)
{
	this->dsv = dsv;
}

void SetTargets::addRenderTargetView(ScriptRenderTargetViewVariable* rtv)
{
	rtvs.push_back(rtv);
}

void SetTargets::execute(const TaskContext& context)
{
	ID3D10DepthStencilView* adsv = dsv?dsv->getDepthStencilView():NULL;
	std::vector<ScriptRenderTargetViewVariable*>::iterator i = rtvs.begin();
	unsigned int nRenderTargets = 0;
	ID3D10RenderTargetView* artvs[32];
	while(i != rtvs.end())
	{
		if(*i)
			artvs[nRenderTargets] = (*i)->getRenderTargetView();
		else
			artvs[nRenderTargets] = NULL;
		nRenderTargets++;
		i++;
	}
	if(nRenderTargets > 0 && rtvs[0])
	{
		D3D10_RENDER_TARGET_VIEW_DESC rtvDesc;
		artvs[0]->GetDesc(&rtvDesc);
		if(rtvDesc.ViewDimension == D3D10_RTV_DIMENSION_TEXTURE2D)
		{
			ID3D10Texture2D* texture;
			artvs[0]->GetResource((ID3D10Resource**)&texture);
			D3D10_TEXTURE2D_DESC texDesc;
			texture->GetDesc(&texDesc);
			D3D10_VIEWPORT vp;
			vp.Height = texDesc.Height;
			vp.Width = texDesc.Width;
			vp.TopLeftX = 0;
			vp.TopLeftY = 0;
			vp.MinDepth = 0;
			vp.MaxDepth = 1;
			context.theatre->getDevice()->RSSetViewports(1, &vp);
			texture->Release();
		}
	}
	context.theatre->getDevice()->OMSetRenderTargets(nRenderTargets, artvs, adsv);
}

