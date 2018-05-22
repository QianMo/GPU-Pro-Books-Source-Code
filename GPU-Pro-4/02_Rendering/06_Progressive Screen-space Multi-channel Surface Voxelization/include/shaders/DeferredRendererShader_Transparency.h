#ifndef _DR_SHADER_TRANSPARENCY_
#define _DR_SHADER_TRANSPARENCY_

#include "DeferredRendererShader.h"

class DRShaderTransparency : public DRShader
{
private:
	GLint  uniform_transparency_width,
		   uniform_transparency_height,
		   uniform_transparency_light_pos,
		   uniform_transparency_eye_pos,
		   uniform_transparency_light_dir,
		   uniform_transparency_light_col,
		   uniform_transparency_light_attn,
		   uniform_transparency_light_active,
		   uniform_transparency_light_range,
		   uniform_transparency_MVP_inverse,
		   uniform_transparency_M_light,
		   uniform_transparency_light_size,
		   uniform_transparency_use_shadow,
		   uniform_transparency_RT_shadow,
		   uniform_transparency_RT_specular,
		   uniform_transparency_RT_depth,
		   uniform_transparency_num_lights,
		   uniform_transparency_texture1,
    	   uniform_transparency_texture2,
		   uniform_transparency_ambient_term;

	static char DRSH_Transparency_Vertex[],
		        DRSH_Transparency_Fragment_Header[],
				DRSH_Transparency_Fragment_Core[];
	
	bool initialized;
	int shadow_method;
	float eye_pos[3];
	float fmat_MVP_inv[16], fmat_L[16];
	class DRLight *L;

public:
	DRShaderTransparency();
	virtual void start();
	virtual bool init(class DeferredRenderer* _renderer);
	void setModelViewMatrixInverse(const GLdouble* mat){for (int j=0;j<16;j++) fmat_MVP_inv[j] = (float)mat[j];}
	void setLightMatrix(const GLdouble* mat){for (int j=0;j<16;j++) fmat_L[j] = (float)mat[j];}
	void setCurrentLight(class DRLight *light) {L = light;}
	void setEyePosition(float * eye) {eye_pos[0]=eye[0]; eye_pos[1] = eye[1]; eye_pos[2] = eye[2];}
};


#endif