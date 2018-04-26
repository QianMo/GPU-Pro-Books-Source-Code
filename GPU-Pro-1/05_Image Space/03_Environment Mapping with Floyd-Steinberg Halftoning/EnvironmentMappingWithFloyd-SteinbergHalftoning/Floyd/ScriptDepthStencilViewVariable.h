#pragma once
#include "scriptvariable.h"

class ScriptDepthStencilViewVariable :
	public ScriptVariable
{
	ID3D10DepthStencilView* depthStencilView;
public:
	ScriptDepthStencilViewVariable(ID3D10DepthStencilView* depthStencilView)
	{
		this->depthStencilView = depthStencilView;
	}
	~ScriptDepthStencilViewVariable(void);

	ID3D10DepthStencilView* getDepthStencilView() {return depthStencilView;}
	void setDepthStencilView(ID3D10DepthStencilView* depthStencilView) {this->depthStencilView = depthStencilView;}

	void releaseResource(){ depthStencilView->Release(); depthStencilView=NULL;}
};
