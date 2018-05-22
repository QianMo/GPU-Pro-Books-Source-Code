#ifndef _DR_SHADER_GI_IV_PROPAGATION_
#define _DR_SHADER_GI_IV_PROPAGATION_

#include "DeferredRendererShader_GI.h"
#include "BBox3D.h"

class DRShaderGI_IV_Propagation : public DRShaderGI
{
protected:
	GLint   uniform_photonmap_composited_red,
			uniform_photonmap_composited_green,
			uniform_photonmap_composited_blue,
			uniform_photonmap_occupied,
			uniform_iteration,
			uniform_cfactor,
			uniform_photonmap_resolution;

	int iteration;
	float cfactor;
	int dimx, dimy, dimz;
	static char DRSH_GI_IV_Vert[],
				DRSH_GI_IV_Frag[];

public:
	virtual void start();
	virtual bool init(class DeferredRenderer* _renderer);
	void setIteration(int i) {iteration = i;}
	void setSpreadFactor(float s) {cfactor = s;}
	void setDimensions(int x, int y, int z) {dimx=x; dimy=y; dimz=z;}
};

#endif


