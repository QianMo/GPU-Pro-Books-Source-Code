#ifndef _DR_SHADER_VIEWPHOTONMAP_
#define _DR_SHADER_VIEWPHOTONMAP_

#include "DeferredRendererShader.h"
#include "GlobalIlluminationRenderer.h"

class DRShaderViewPhotonMap : public DRShader
{
private:
	GLint uniform_viewphotonmap_shR_buffer,
		  uniform_viewphotonmap_shG_buffer,
		  uniform_viewphotonmap_shB_buffer,
		  uniform_viewphotonmap_buffer_negative;

	static char DRSH_ViewPhotonMap_Vertex[],
		        DRSH_ViewPhotonMap_Fragment[];

	bool initialized;

public:
	DRShaderViewPhotonMap();
	virtual void start();
	virtual void stop();
	virtual bool init(class DeferredRenderer* _renderer);
};

#endif
