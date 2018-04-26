#pragma once

class ScriptVariable
{

public:
	ScriptVariable(void)
	{
	}

	virtual ~ScriptVariable(void)
	{
	}

	virtual void releaseResource()=0;
};
