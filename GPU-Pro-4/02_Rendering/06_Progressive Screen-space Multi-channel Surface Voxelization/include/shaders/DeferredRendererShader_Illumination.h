#ifndef _DR_SHADER_ILLUMINATION_
#define _DR_SHADER_ILLUMINATION_

#include "DeferredRendererShader.h"

class DRShaderIllumination : public DRShader
{
private:
	GLint  uniform_illumination_width,
		   uniform_illumination_height,
		   uniform_illumination_light_pos,
		   uniform_illumination_eye_pos,
		   uniform_illumination_light_dir,
		   uniform_illumination_light_col,
		   uniform_illumination_light_attn,
		   uniform_illumination_light_active,
		   uniform_illumination_light_range,
		   uniform_illumination_RT_normals,
		   uniform_illumination_RT_depth,
		   uniform_illumination_RT_specular,
		   uniform_illumination_MVP_inverse,
		   uniform_illumination_Projection,
		   uniform_illumination_M_light,
		   uniform_illumination_light_size,
		   uniform_illumination_use_shadow,
		   uniform_illumination_RT_shadow,
		   uniform_illumination_noise,
		   uniform_illumination_shadow_size,
	       uniform_illumination_is_cone,
		   uniform_illumination_cone;

	static char DRSH_Vertex[],
		        DRSH_Illumination_Fragment_Header[],
				DRSH_Illumination_Fragment_Core[],
		        DRSH_Illumination_Fragment_Footer[];

	int shadow_method;
	float fmat_MVP_inv[16], fmat_P[16], fmat_L[16];
	float eye_pos[3];
	class DRLight *L;

public:
	virtual void start();
	virtual bool init(class DeferredRenderer* _renderer);
	void setModelViewMatrixInverse(const GLdouble* mat){for (int j=0;j<16;j++) fmat_MVP_inv[j] = (float)mat[j];}
	void setProjectionMatrix(const GLdouble* mat){for (int j=0;j<16;j++) fmat_P[j] = (float)mat[j];}
	void setLightMatrix(const GLdouble* mat){for (int j=0;j<16;j++) fmat_L[j] = (float)mat[j];}
	void setCurrentLight(class DRLight *light) {L = light;}
	void setEyePosition(float * eye) {eye_pos[0]=eye[0]; eye_pos[1] = eye[1]; eye_pos[2] = eye[2];}
};

#endif
