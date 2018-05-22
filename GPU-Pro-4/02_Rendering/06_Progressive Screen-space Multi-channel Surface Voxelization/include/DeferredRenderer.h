//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//  EaZD Deferred Renderer                                                  //
//  Georgios Papaioannou, 2009                                              //
//                                                                          //
//  This is a free deferred renderer. The library and the source            //
//  code are free. If you use this code as is or any part of it in any kind //
//  of project or product, please acknowledge the source and its author.    //
//                                                                          //
//  For manuals, help and instructions, please visit:                       //
//  http://graphics.cs.aueb.gr/graphics/                                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

#ifndef _DEFERRED_RENDERER
#define _DEFERRED_RENDERER

#define DR_VERSION 1
#define DR_REVISION 5

#define USING_GLUT

#define _DISABLE_GLSL_LOGGING_

#pragma once
#include "glsl.h"

#include "cglib.h"
#include "DeferredRendererShaders.h"
#include "GlobalIlluminationRenderer.h"
#include <set>

using namespace cwc;
using namespace std;

#define DR_BUFFER_ALBEDO       1
#define DR_BUFFER_CSS_DEPTH    2
#define DR_BUFFER_ECS_DEPTH    4
#define DR_BUFFER_CSS_NORMAL   8
#define DR_BUFFER_ECS_NORMAL  16
#define DR_BUFFER_UV1         32
#define DR_BUFFER_UV2         64
#define DR_BUFFER_SHADOWMAP  128
#define DR_BUFFER_LIGHTS     256
#define DR_BUFFER_SPECULAR   512
#define DR_BUFFER_GLOW      1024

#define DR_ERROR_NONE                       0
#define DR_ERROR_INSUFFICIENT_BUFFERS       1
#define DR_ERROR_UNINITIALIZED              2
#define DR_ERROR_BAD_FBO                    3
#define DR_ERROR_NOT_SUPPORTED              4
#define DR_ERROR_INSUFFICIENT_TEXTURE_UNITS 5

#define DR_TARGET_COLOR        0
#define DR_TARGET_NORMAL       1
#define DR_TARGET_SPECULAR     2
#define DR_TARGET_LIGHTING     3
#define DR_TARGET_DEPTH        4
#define DR_TARGET_FRAMEBUFFER  5
#define DR_TARGET_SHADOWMAP    6
// #define DR_TARGET_SHADOWMAP_COLOR   6
// #define DR_TARGET_SHADOWMAP_NORMAL  6
#define DR_TARGET_GLOW         7
#define DR_TARGET_AO           8
#define DR_TARGET_VBO          9

#define DR_SHADOW_GAUSS        3

#define DR_GI_METHOD_NONE      0 // default
#define DR_GI_METHOD_IV        5

#define DR_HDR_MANUAL          0 // default
#define DR_HDR_AUTO            1

#define DR_NUM_LIGHTS  128
#define DR_LIGHT_OMNI 1
#define DR_LIGHT_SPOT 0

#define DR_AMBIENT_BLEND_MODULATE 0
#define DR_AMBIENT_BLEND_ADD      1

float getHighResTime(); 

GLboolean invert_matrix(const GLdouble * m, GLdouble * out);
void matmul(GLdouble * product, const GLdouble * a, const GLdouble * b);

class DRShadowFBO
{
private:
	static GLuint fbo;
	
public:
	DRShadowFBO(){};
	GLuint getInstance();
	void release();
};

class DRLight
{
protected:
	GLdouble matrix_shadow_modelview[16];
	GLdouble matrix_shadow_projection[16];
    GLfloat **OmniMVP,**OmniMod;
	GLuint uniform_primary_light;
	int active;
	int shadow_active;
	float intensity; // can be more than 1
	float color[3];
	float ambient[3];
	float light_pos[3];
	float light_tgt[3];
	float light_pos_transformed[3];
	float light_tgt_transformed[3];
	float size;
	float light_near,
		  light_far,
		  light_near_transformed,
		  light_far_transformed,
          apperture;
	bool  is_attenuating,
		  is_cone,
	      casts_shadows;
	int   shadow_method;
	int   shadow_res;
	GLuint shadow_map;
	GLuint color_map;
	GLuint normal_map;
	GLuint shadow_FBO;
	bool  update_shadow_map;
	bool  gi_enabled,
	      has_extended_data; // normal/depth/diffuse buffer storage instead of just depth
	int skip_frames, frame_loop;
	
	void transform(); // use the scene graph equiv. light node transformation passed as data.
	
public:
	DRLight();
	~DRLight();
	void setPosition(float lpx, float lpy, float lpz);
	float* getPosition() {return light_pos;}
	float* getTransformedPosition() {return light_pos_transformed;}
	void setTarget(float ltx, float lty, float ltz);
	float* getTarget() {return light_tgt;}
	float* getTransformedTarget() {return light_tgt_transformed;}
	float  getTransformedFar() {return light_far_transformed;}
	float  getTransformedNear() {return light_near_transformed;}
	void setRanges(float near, float far);
	void setColor(float r, float g, float b);
	void setAmbient(float r, float g, float b);
	void setIntensity(float bright);
	void setSize(float sz) {size = sz;}
	float getSize() {return size;}
	float getIntensity(){return intensity;}
	void setAttenuation(bool attn);
	void setCone(float a) {apperture=a/2.0f;}
	float getCone() {return 2.0f*apperture;}
	float getApperture() {return apperture;}
	void enable(bool en);
	void enableExtendedData(bool e);
	void enableShadows(bool shadows);
	void setShadowResolution(unsigned int res);
	void setShadowSamplingMethod(int method);
	void setShadowFBO(GLuint fbo);
	void skipFrames(int s) {skip_frames=s;}
	int isActive() {return active;}
	int isAttenuated() {return (int)is_attenuating;}
	int isShadowEnabled() {return shadow_active;}
	bool hasExtendedData() {return has_extended_data;}
	bool isConical() {return is_cone;}
	void setConical(bool c) {is_cone=c;}
	bool isGIEnabled() {return gi_enabled;}
	void enableGI(bool g) {gi_enabled=g;}
	bool needsUpdate() {return update_shadow_map&(frame_loop==0);}
	int getShadowMap() {return shadow_map;}
	int getShadowMapRes() {return shadow_res;}
	int getShadowMapColorBuffer() {return color_map;}
	int getShadowMapNormalBuffer() {return normal_map;}
	GLdouble * getProjectionMatrix(){return matrix_shadow_projection;}
	GLdouble * getModelviewMatrix(){return matrix_shadow_modelview;}
	float * getColor() {return color;}
	void setupShadowMap();
	void update() {update_shadow_map=true; transform();}
	void * data; //reference to external data. DO NOT ALLOCATE/DEALLOCATE
};


class GenericRenderer
{
protected:
	class World3D *root;				// Attached scene graph. If !NULL, overrides
	                                    // draw callbacks below
	void (*draw_callback)();			// callback of user-defined scene draw function
	void (*camera_callback)();			// callback of user-defined camera tranformation
	
	float global_ambient[3];            // global scene ambient color. Affects ambient occlusion
	int width,							// window width and height
		height;
	DRLight lights[DR_NUM_LIGHTS];   	// array of available DR lights (max 64)
	int num_lights;						// current num. of lights set
	float background[3];
	
	double timer_total, counter_total;  // internal statistics
#ifndef WIN32
	struct timeval tp;
#endif

	Timer *t_g_buffer;

public:
	virtual int init() {return 0;}
	virtual int init(int request_mask) {return 0;}
	virtual void draw() {}
	virtual void showStatistics() {}
	float getFrameRate() {return 1000.0f/(float)timer_total;}
	virtual void setDrawCallback(void (*drw)() ) {}
	void setSceneRoot(class World3D *nd ) {root = nd;}
	class World3D * getSceneRoot() {return root;}
	
	virtual void setCameraCallback(void (*cm)() ) {}
	virtual void resize(int w, int h) {width=w; height=h;}
	void setAmbient(float r, float g, float b) {global_ambient[0]=r; global_ambient[1]=g; global_ambient[2]=b; }
	void getAmbient(float *_r, float *_g, float *_b) {*(_r)=global_ambient[0];*(_g)=global_ambient[1];*(_b)=global_ambient[2];}
	virtual int createLight()=0;
	virtual void setLightRanges(int light, float near, float far)=0;
	virtual void setLightColor(int light, float r, float g, float b)=0;
	virtual void setLightAmbient(int light, float r, float g, float b)=0;
	virtual void setLightPosition(int light, float x, float y, float z)=0;
	virtual void setLightTarget(int light, float x, float y, float z)=0;
	virtual void setLightIntensity(int light, int intens)=0;
	virtual void enableLight(int light, bool ltenable)=0;
	virtual void enableLightShadows(int light, bool sh)=0;
	virtual void setLightAttenuation(int light, bool attn)=0;
	virtual void setLightSize(int light, float sz)=0;
	virtual void setLightSkipFrames(int light, int sf)=0;
	virtual void setLightCone(int light, float a)=0;
	virtual void setDOF(float distance, float range)=0;
	virtual void setShadowResolution(int light, int res)=0;
	void setBackground(float r, float g, float b) {background[0]=r;background[1]=g;background[2]=b;}
	void getBackground(float *r, float *g, float *b) {*r=background[0]; *g=background[1]; *b=background[2];}
	virtual void attachLightData(int light, void *dt);
	DRLight *getLight(int i) {if (i<num_lights && i>=0) return &(lights[i]); else return NULL;}
	DRLight *getLights() {return lights;}
	int getNumLights() {return num_lights;}
	
};

class DeferredRenderer: public GenericRenderer
{
private:
	int buffer_bits;					// buffer enable mask
	GLint maxbuffers;					// max num of system render buffers
	GLuint FBO;							// internal FBO for multiple targets
	GLuint glow_FBO;					// internal FBO for glow effects
	GLuint AO_FBO;						// frame buffer object for storing ambient occlusion (low res)
	GLuint framebuffer_FBO;	     		// internal FBO for final image and post effects. Can be different size than FBO
	DRShadowFBO shadow_fbo;				// common shadow frame buffer object for all lights
	DRShaderMRT shader_MRT;				// multiple render target shader
	GLuint buffer[9];					// array of allocated buffers (texture IDs)
	GLenum multipleRenderTargets[4];	// holds the rendering buffers
	GLuint noise;						// noise texture id
    
	glShaderManager SM;					// global shader manager object
	DRShaderFrameBuffer shader_FB;   	// shader for producing the final frame buffer
	DRShaderIllumination shader_Lighting; // lighting processing shader
    DRShaderPost shader_PostProcess; 	// shader for post-processing and showing the frame buffer
	DRShaderTransparency shader_Trans;	// shader for transparent surfaces (non screen-space pass)
	DRShaderGlow shader_Glow;           // special fx buffer shader
	DRShaderClearMRT shader_ClearMRT;   // shader for cleaning the MRTs
	DRShaderShadowMap shader_ShadowMap; // shader for writing extended shadow map data
	DRShaderViewDepthBuffer shader_ViewDepthBuffer; // shader for viewing the linearized DepthBuffer
	DRShaderViewPhotonMap shader_ViewPhotonMap; // shader for viewing the photon map
	
	GlobalIlluminationRenderer * gi_renderer; // renderer for GI effects
	
	float focal_distance,				// depth of field focal distance
		  focal_range;                  // depth of field focal range (fd+/-fr comes in focus)
	float units_per_meter;				// number of logical (WCS) units per meter
	float buffer_scale;					// forced scaling of buffer (parameter)
	int	actual_width,					// internal buffer width and height
		actual_height;

	bool is_fixed_size,					// flag for buffer scaling disable
		 is_initialized,				// if buffer ok this is set
		 use_gi,			            // enable global illumination
	     force_update_shadows;			// forced update of all shadow maps

    GLdouble matrix_MV[16];				// stores the camera matrix
    GLdouble matrix_MV_inverse[16];		// stores the inverse camera matrix
    GLdouble matrix_P[16];	           	// stores the projection matrix
    GLdouble matrix_P_inverse[16];		// stores the inverse projection matrix
    GLdouble matrix_MVP[16];			// stores the camera and projection matrix
    GLdouble matrix_MVP_inverse[16];	// stores the inverse camera and projection matrix
	float eye[4];                       // world CS eye coords
	bool calc_matrix;					// MVP_inverse requires recalculation
	int shadow_method;                  // type of sampling to use for shadow maps
	BBox3D bbox;                        // scene bounding box - if any.
	
	int hdr_method;                     // manual HDR settings (default) or auto.
	Vector3D hdr_white_point;           // The white color (HDR) - Values beyond this triplet
	                                    // are saturated.
	float hdr_key;                      // The desired luminance of the scene.

	float light_array_pos[3*DR_NUM_LIGHTS]; // temporary transformed lights to be passed to shaders
	float light_array_dir[3*DR_NUM_LIGHTS]; //
	float light_array_col[3*DR_NUM_LIGHTS]; //
	int   light_array_attn[DR_NUM_LIGHTS];  //
	int   light_array_active[DR_NUM_LIGHTS];//
	float light_array_range[DR_NUM_LIGHTS]; //
	GLdouble light_array_matrix[DR_NUM_LIGHTS][16];

	int volumebuffer_resolution;        // max dimension of volume buffer
	
	int gi_method;						// method to use. Default is none.
	float ao_buffer_ratio;              // controls the size of the AO buffer relative to
	                                    // the primary buffer
	int ambient_blending;               // ambient color blending mode. Set according to GI/AO method.
	                                    // Used by framebuffer shader.
	
	void buildShaders();				// dynamically configures and compiles shaders
	int resizeBuffers();				// resets the DR fbo to apropriate dimensions
	void drawShadowMaps();
	void drawMultipleRenderTargets();
	void drawLighting();
	void drawTransparency();
	void drawGlobalIllumination();
	void drawFrameBuffer();
	void drawAmbientOcclusionSSAO();
	void finalRender();
	void transformCoordinates();
	void statistics();
	
	int    frames;
	
public:
	// Construction and initialization methods
	DeferredRenderer();
	virtual ~DeferredRenderer();
	virtual int init();
	virtual int init(int request_mask);
	void initLighting();
	void rebuildShaders(void) {buildShaders();}

	// Buffer-related methods
	void enableBuffer(int request_mask);
	void disableBuffer(int request_mask);
	virtual void resize(int w, int h);
	int getWidth(void) { return width; }
	int getHeight(void) { return height; }
	int getActualWidth(void) { return actual_width; }
	int getActualHeight(void) { return actual_height; }
	void setFixedBufferSize(int width, int height);
	void setFreeBufferSize();
	void setBufferScale(float scale);
	GLuint getBuffer(int id);
	void show(int target);
	virtual void showStatistics();
	
	// Draw methods and callback registration
	virtual void setDrawCallback(void (*drw)() );
	virtual void setCameraCallback(void (*cm)() );
	virtual void draw();
	
	// Ambient occlusion methods
	void setAOBufferRatio(float r) {ao_buffer_ratio=r; resizeBuffers();}
	GLuint getAOfbo() {return AO_FBO;}
	
	// High dynamic range rendering methods
	void setHDRKey(float k) {if (k>0.1f) hdr_key=k;}
	float getHDRKey() {return hdr_key;}
	void setHDRWhitePoint(float wr, float wg, float wb) {if (wr+wb+wg>0.0f) hdr_white_point = Vector3D(wr,wg,wb);}
	Vector3D getHDRWhitePoint() {return hdr_white_point;}
	void setHDRMethod(int m) {(m==DR_HDR_AUTO)?hdr_method=m:hdr_method=DR_HDR_MANUAL;buildShaders();}
	int getHDRMethod() {return hdr_method;}
	
	// Skymodel methods
	int getAmbientBlendMode(){return ambient_blending;}
	
	// Light sources management methods
	virtual int createLight();
	virtual void setLightRanges(int light, float near, float far);
	virtual void setLightColor(int light, float r, float g, float b);
	virtual void setLightAmbient(int light, float r, float g, float b);
	virtual void setLightPosition(int light, float x, float y, float z);
	virtual void setLightTarget(int light, float x, float y, float z);
	virtual void setLightIntensity(int light, int intens);
	virtual void enableLight(int light, bool ltenable);
	virtual void enableLightShadows(int light, bool sh);
	virtual void setLightAttenuation(int light, bool attn);
	virtual void setLightSize(int light, float sz);
	virtual void setLightCone(int light, float a);
	virtual void setDOF(float distance, float range);
	virtual void setShadowResolution(int light, int res);
	virtual void setLightSkipFrames(int light, int sf);
	void setShadowMethod(int method);
	int getShadowMethod(){return shadow_method;}
	GLdouble * getLightMatrix(int light) {return &(light_array_matrix[light][0]);}
	
	// Geometric data methods
	float * getEyePos() {return eye;}
	BBox3D getBoundingBox() {return bbox;}
	
	// Global illumination methods
	void  enableGI(bool e) {use_gi = e;}
	bool  isGIEnabled() {return use_gi;}
	void  setGIMethod(int gm) {gi_method=gm;}
	GlobalIlluminationRenderer * getGIRenderer() {return gi_renderer;}
	
	// Shader data access methods
	glShaderManager getShaderManager() {return SM;}
	GLuint getNoiseTexture(){return noise;}
	
	// Volume buffer methods
	void  setVolumeBufferResolution(int r) {volumebuffer_resolution=(r>4?r:4);}
	int   getVolumeBufferResolution() {return volumebuffer_resolution;}
	
	// Viewing transformation methods
	GLdouble * getModelViewMatrix() {return matrix_MV;}
	GLdouble * getModelViewMatrixInv() {return matrix_MV_inverse;}
	GLdouble * getModelViewProjectionMatrix() {return matrix_MVP;}
	GLdouble * getModelViewProjectionMatrixInv() {return matrix_MVP_inverse;}
	GLdouble * getProjectionMatrix() {return matrix_P;}
	GLdouble * getProjectionMatrixInv() {return matrix_P_inverse;}
};

#endif
