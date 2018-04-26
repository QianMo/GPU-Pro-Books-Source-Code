#pragma once
#include "scriptvariable.h"

class ScriptResourceVariable :
	public ScriptVariable
{
	ID3D10Resource* resource;
public:
	ScriptResourceVariable(ID3D10Resource* resource)
	{this->resource = resource;}
	~ScriptResourceVariable(void);
	ID3D10Resource* getResource(){return resource;}
	void setResource(ID3D10Resource* resource) {this->resource = resource;}
	void releaseResource(){ resource->Release(); resource=NULL;}
};