#ifndef _DR_RENDERER_GI_IV_
#define _DR_RENDERER_GI_IV_

#include "GlobalIlluminationRenderer.h"
#include "shaders/DeferredRendererShader_GI_IV_Injection_SS.h"
#include "shaders/DeferredRendererShader_GI_IV_Injection_SS_Cleanup.h"
#include "shaders/DeferredRendererShader_GI_IV_Propagation.h"
#include "FrameBufferObject.h"
#include "Timer.h"

class GlobalIlluminationRendererIV : public GlobalIlluminationRenderer
{
protected:
	DRShaderGI_IV_Propagation * propagation_shader;
	DRShaderGI_IV_Injection_SS   * injection_SS_shader;		// screen space voxelization
	DRShaderGI_IV_Injection_SS_Cleanup   * injection_SS_Cleanup_shader;	// screen space voxelization
	// plus the inherited shader var (for GI fb rendering) of type DRShaderGI_IV

	int resolution;
	unsigned int noise;
	bool params_decoded;
	int inj_width, inj_height, inj_grid_size;
	bool inj_camera, inj_lights;
	float cfactor;
	bool write_debug_data;

	unsigned int propagateTexId[6], propagateFboId[2];
	unsigned int accumulatorTexId[3];
	unsigned int vplTexId[3];
	unsigned int occupiedTexId;  // holds the occupancy of a voxel.
	unsigned int tmpTexId[3];
	unsigned int fbo;            // fbo: z-direction of photonmap 3d volume
	unsigned int photonMapTexId[3];
    unsigned int inject_SSTexId[2][4];
    FramebufferObject * inject_SSFboId[2];

	unsigned int read_tex, write_tex;
	unsigned int inject_tex, clean_tex;
	unsigned int mrts[16];

	unsigned int m_width, m_height, m_depth;
	GLenum  m_format;
	int     propagation_steps;
	GLuint vboID;
	float *fbVerts;
	float transform[16];
	BBox3D m_bbox;

	// methods
	void buildInjectionGrid();
	void drawInjectionGrid();

public:
	GlobalIlluminationRendererIV();
	~GlobalIlluminationRendererIV();
	virtual bool init(class DeferredRenderer * renderer);
	bool createTexture3D();
	void clearTexture3D();
	void drawTexture3D(GLint texId);
	virtual void draw();
	int getNormalBuffer()	{return inject_SSTexId[clean_tex][0];}	// <-- 0 signifies the normals buffer
	int getSHRBuffer()	{return inject_SSTexId[clean_tex][1];}
	int getSHGBuffer()	{return inject_SSTexId[clean_tex][2];}
	int getSHBBuffer()	{return inject_SSTexId[clean_tex][3];}
	BBox3D getBBox() { return m_bbox; }

	void setInjectCameraLights (bool c, bool l) { inj_camera = c; inj_lights = l; }
	bool getInjectCamera (void) { return inj_camera; }
	bool getInjectLights (void) { return inj_lights; }

	unsigned int getSizeData ();
	float * getData (GLint texId);

	Timer *t_inc_light_injection, *t_inc_camera_injection;
	Timer *t_inc_light_cleanup,   *t_inc_camera_cleanup;
};

#endif
