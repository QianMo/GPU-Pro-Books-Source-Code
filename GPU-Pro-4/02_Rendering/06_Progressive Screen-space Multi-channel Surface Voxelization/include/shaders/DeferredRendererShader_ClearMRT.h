#ifndef _DR_SHADER_CLEARMRT_
#define _DR_SHADER_CLEARMRT_


#include "DeferredRendererShader.h"

class DRShaderClearMRT : public DRShader
{
private:
	GLint uniform_clear_background;
	
	static char DRSH_Vertex[],
		        DRSH_Fragment_Clear[];
	bool initialized;

public:
	DRShaderClearMRT();
	virtual void start();
	virtual bool init(class DeferredRenderer* _renderer);
};


#endif