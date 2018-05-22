#ifndef _DR_SHADER_VIEWDEPTHBUFFER_
#define _DR_SHADER_VIEWDEPTHBUFFER_

#include "DeferredRendererShader.h"

class DRShaderViewDepthBuffer : public DRShader
{
private:
	GLint uniform_viewdepthbuffer_buffer,
		  uniform_viewdepthbuffer_zNear,
		  uniform_viewdepthbuffer_zFar;

	static char DRSH_ViewDepthBuffer_Vertex[],
		        DRSH_ViewDepthBuffer_Fragment[];

	bool initialized;

public:
	DRShaderViewDepthBuffer();
	virtual void start();
	virtual void stop();
	virtual bool init(class DeferredRenderer* _renderer);
};

#endif
