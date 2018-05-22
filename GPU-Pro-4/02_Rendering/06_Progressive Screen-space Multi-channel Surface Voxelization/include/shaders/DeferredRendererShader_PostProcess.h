#ifndef _DR_SHADER_POSTPROCESS_
#define _DR_SHADER_POSTPROCESS_

#include "DeferredRendererShader.h"

class DRShaderPost : public DRShader
{
private:
	GLint  uniform_postprocess_RT_lighting,
		   uniform_postprocess_framebuffer,
		   uniform_postprocess_depth,
		   uniform_postprocess_height,
		   uniform_postprocess_width,
		   uniform_postprocess_P_inv,
		   uniform_postprocess_hdr_key,
		   uniform_postprocess_hdr_white;

	static char DRSH_Vertex[],
		        DRSH_Postprocess_Fragment_Header[],
				DRSH_Postprocess_Fragment_Color[],
				DRSH_Postprocess_Fragment_ToneMapping[],
				DRSH_Postprocess_Fragment_Footer[];
public:
	static char DRSH_Postprocess_ToneMapping_Auto[],
				DRSH_Postprocess_ToneMapping_Manual[];
	
private:
	int hdr_method;
	bool initialized;
	float fmat_P_inv[16];

public:
	DRShaderPost();
	virtual void start();
	virtual bool init(class DeferredRenderer* _renderer);
	void setProjectionMatrixInverse(const GLdouble* mat){for (int j=0;j<16;j++) fmat_P_inv[j] = (float)mat[j];}
};

#endif
