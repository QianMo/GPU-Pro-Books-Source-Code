#ifndef _DR_SHADER_SHADOWMAP_
#define _DR_SHADER_SHADOWMAP_

#include "DeferredRendererShader.h"

class DRShaderShadowMap : public DRShader
{
private:
	GLint uniform_texture1,
		  uniform_emission,
		  uniform_light_color,
		  uniform_attenuating,
		  uniform_far,
		  uniform_cone, 
		  uniform_use_cone;
		  
	static char DRSH_Vertex[],
		        DRSH_Fragment[];
	bool initialized;
	class DRLight *L;

public:
	DRShaderShadowMap();
	virtual ~DRShaderShadowMap();
	virtual void start();
	virtual bool init(class DeferredRenderer* _renderer);
	void setCurrentLight(class DRLight *light) {L = light;}
};

#endif
