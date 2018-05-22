#ifndef _DR_SHADER_GLOW_
#define _DR_SHADER_GLOW_

#include "DeferredRendererShader.h"

class DRShaderGlow : public DRShader
{
private:
	GLint  uniform_glow_RT_lighting,
		   uniform_glow_framebuffer,
		   uniform_glow_height,
		   uniform_glow_width,
		   uniform_glow_hdr_key,
		   uniform_glow_hdr_white;

	static char DRSH_Vertex[],
		        DRSH_Glow_Fragment_Header[],
				DRSH_Glow_Fragment_Core[],
		        DRSH_Glow_Fragment_Footer[];
	
	bool initialized;
	int hdr_method;

public:
	DRShaderGlow();
	virtual void start();
	virtual bool init(class DeferredRenderer* _renderer);
};









#endif
