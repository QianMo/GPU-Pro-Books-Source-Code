#ifndef _DR_SHADER_GI_IV_
#define _DR_SHADER_GI_IV_

#include "DeferredRendererShader_GI.h"
#include "BBox3D.h"

class DRShaderGI_IV : public DRShaderGI
{
protected:
	GLint  uniform_width,
		   uniform_height,
		   uniform_RT_normals,
		   uniform_RT_depth,
		   uniform_Noise,
		   uniform_photonmap_red,
		   uniform_photonmap_green,
		   uniform_photonmap_blue,
		   uniform_MVP_inverse,
		   uniform_MVP,
		   uniform_Projection_inverse,
		   uniform_Projection,
		   uniform_R_wcs,
		   uniform_factor,
		   uniform_voxelsize,
		   uniform_samples,
		   uniform_photonmap_res;

	int    dim[3];
	BBox3D bbox;
		   
	static char DRSH_GI_IV_Vert[],
				DRSH_GI_IV_Frag_Header[],
				DRSH_GI_IV_Frag_Main[];
public:
	static char DRSH_GI_IV_Frag_SH[];

	virtual void start();
	virtual bool init(class DeferredRenderer* _renderer);
	void setDimensions(int x, int y, int z) {dim[0] = x; dim[1]=y; dim[2]=z;}
	void setBoundingBox(BBox3D b) {bbox = b;} 
};

#endif
