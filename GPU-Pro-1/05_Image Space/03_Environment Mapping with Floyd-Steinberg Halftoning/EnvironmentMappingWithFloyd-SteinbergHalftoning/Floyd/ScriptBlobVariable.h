#pragma once
#include "scriptvariable.h"

class ScriptBlobVariable :
	public ScriptVariable
{
	void* data;
	unsigned int nDataBytes;
public:
	ScriptBlobVariable(void* data, unsigned int nDataBytes)
	{
		this->data = data;
		this->nDataBytes = nDataBytes;
	}
	~ScriptBlobVariable(void);

	void* getPointer(){return data;}
	unsigned int getByteCount(){return nDataBytes;}

	void setData(void* data, unsigned int nDataBytes)
	{
		this->data = data;
		this->nDataBytes = nDataBytes;
	}

	void releaseResource(){ delete data;}
};
