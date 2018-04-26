#pragma once
#include "scriptvariable.h"

class ScriptRenderTargetViewVariable :
	public ScriptVariable
{
	ID3D10RenderTargetView* renderTargetView;
public:
	ScriptRenderTargetViewVariable(ID3D10RenderTargetView* renderTargetView)
	{
		this->renderTargetView = renderTargetView;
	}
	~ScriptRenderTargetViewVariable(void);
	ID3D10RenderTargetView* getRenderTargetView(){return renderTargetView;}
	void setRenderTargetView(ID3D10RenderTargetView* renderTargetView) {this->renderTargetView = renderTargetView;}

	void releaseResource(){ renderTargetView->Release(); renderTargetView=NULL;}
};
