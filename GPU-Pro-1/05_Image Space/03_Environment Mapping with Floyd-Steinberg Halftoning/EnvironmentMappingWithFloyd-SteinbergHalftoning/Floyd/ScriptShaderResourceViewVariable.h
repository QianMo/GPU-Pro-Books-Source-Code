#pragma once
#include "scriptvariable.h"

class ScriptShaderResourceViewVariable :
	public ScriptVariable
{
	ID3D10ShaderResourceView* shaderResourceView;
public:
	ScriptShaderResourceViewVariable(ID3D10ShaderResourceView* shaderResourceView)
	{
		this->shaderResourceView = shaderResourceView;
	}
	~ScriptShaderResourceViewVariable(void);
	ID3D10ShaderResourceView* getShaderResourceView(){return shaderResourceView;}
	void setShaderResourceView(ID3D10ShaderResourceView* resource){this->shaderResourceView = resource;}
	
	void releaseResource(){ shaderResourceView->Release(); shaderResourceView=NULL;}
};
