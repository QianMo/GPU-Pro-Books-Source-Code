#ifndef _DR_SHADER_GI_IV_INJECTION_SS_CLEANUP_
#define _DR_SHADER_GI_IV_INJECTION_SS_CLEANUP_

#include "DeferredRendererShader_GI.h"
#include <Vector3D.h>

class DRShaderGI_IV_Injection_SS_Cleanup : public DRShaderGI
{
protected:
	GLint	uniform_cln_vol_width, uniform_cln_vol_height,
			uniform_cln_prebaked_lighting, uniform_cop,
			uniform_cln_voxel_radius, uniform_cln_voxel_half_size,
			uniform_cln_zbuffer, uniform_cln_albedo, uniform_cln_normals, uniform_cln_lighting, 
			uniform_cln_vol_albedo, uniform_cln_vol_lighting, uniform_cln_vol_color,
			uniform_cln_vol_shR, uniform_cln_vol_shG, uniform_cln_vol_shB, uniform_cln_vol_normals, uniform_cln_vol_zbuffer,
			uniform_cln_P, uniform_cln_P_inverse, uniform_cln_MVP, uniform_cln_MVP_inverse;

	float vol_width, vol_height;
	int prebaked_lighting;
	float voxel_radius;
	float voxel_half_size[3];
	float cop[3];

	static char DRSH_GI_IV_Vert[],
				DRSH_GI_IV_Frag[];

public:
	virtual void start();
	virtual bool init(class DeferredRenderer* _renderer);

	void setVolWidth(float w) {vol_width = w;}
	void setVolHeight(float h) {vol_height = h;}
	void setPrebakedLighting(bool p) {prebaked_lighting = p?1:0;}
	void setVoxelRadius(float r) {voxel_radius = r;}
	void setVoxelHalfSize(Vector3D v) {voxel_half_size[0] = v[0]; voxel_half_size[1] = v[1]; voxel_half_size[2] = v[2];}
	void setCOP(Vector3D v) {cop[0] = v[0]; cop[1] = v[1]; cop[2] = v[2];}
};

#endif
