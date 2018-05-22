#ifndef _DR_SHADER_GI_IV_INJECTION_SS_
#define _DR_SHADER_GI_IV_INJECTION_SS_

#include "DeferredRendererShader_GI.h"

class DRShaderGI_IV_Injection_SS : public DRShaderGI
{
protected:
	GLint	uniform_inj_depth,
			uniform_inj_zbuffer, uniform_inj_albedo, uniform_inj_normals, uniform_inj_lighting,
			uniform_inj_prebaked_lighting,
		//	uniform_inj_MV,
			uniform_inj_P, uniform_inj_MVP_inverse;

	int vol_depth;
	int prebaked_lighting;

	static char DRSH_GI_IV_Vert[],
				DRSH_GI_IV_Geom[],
				DRSH_GI_IV_Frag[];

public:
	virtual void start();
	virtual bool init(class DeferredRenderer* _renderer);

	void setVolDepth(int d) {vol_depth = d;}
	void setPrebakedLighting(bool p) {prebaked_lighting = p?1:0;}
};

#endif
