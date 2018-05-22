#ifndef _DR_SHADER_FRAMEBUFFER_
#define _DR_SHADER_FRAMEBUFFER_


#include "DeferredRendererShader.h"

class DRShaderFrameBuffer : public DRShader
{
private:
	GLint uniform_RT_albedo,
		  uniform_RT_normals,
		  uniform_RT_specular,
		  uniform_RT_lighting,
		  uniform_RT_depth,
		  uniform_RT_ao,
		  uniform_width,
		  uniform_height,
		  uniform_ambient_term,
		  uniform_use_gi,
		  uniform_sun_wcs,
		  uniform_camera_wcs,
		  uniform_unitspm,
		  uniform_MVP_inverse,
		  uniform_MVP,
		  uniform_Projection,
		  uniform_shadow,
		  uniform_L,
		  uniform_show_gi,
		  uniform_noise;

	float fmat_MVP_inv[16], fmat_MVP[16], fmat_P[16], fmat_L[16];
	float light_color[3];
	float light_pos[3];
	float eye_pos[3];
	float units_per_meter;
	bool  show_gi;

	static char DRSH_Vertex[],
		        DRSH_Framebuffer_Fragment_Header[],
				DRSH_Framebuffer_Fragment_Core[],
				DRSH_Framebuffer_Fragment_Core_MultiSampledGI[],
				DRSH_Framebuffer_Fragment_SpecularGI[],
				DRSH_Framebuffer_Fragment_Footer[];

public:
	DRShaderFrameBuffer();
	virtual void start();
	virtual bool init(class DeferredRenderer* _renderer);
	void setModelViewMatrixInverse(const GLdouble* mat){for (int j=0;j<16;j++) fmat_MVP_inv[j] = (float)mat[j];}
	void setModelViewMatrix(const GLdouble* mat){for (int j=0;j<16;j++) fmat_MVP[j] = (float)mat[j];}
	void setProjectionMatrix(const GLdouble* mat){for (int j=0;j<16;j++) fmat_P[j] = (float)mat[j];}
	void setLightMatrix(const GLdouble* mat){for (int j=0;j<16;j++) fmat_L[j] = (float)mat[j];}
	void setLightColor(float r, float g, float b) {light_color[0]=r;light_color[1]=g;light_color[2]=b;}
	void setLightPosition(float x, float y, float z) {light_pos[0]=x; light_pos[1]=y; light_pos[2]=z;}
	void setWorldUnitsPerMeter(float upm) {units_per_meter=upm;}
	void setEyePosition(float x, float y, float z) {eye_pos[0]=x; eye_pos[1]=y; eye_pos[2]=z;}
};

#endif
