#ifndef _DR_GI_RENDERER_
#define _DR_GI_RENDERER_

#include "shaders/DeferredRendererShader_GI.h"
#include <set>

using namespace std;

class GlobalIlluminationRenderer
{
protected:
	class DeferredRenderer * renderer;
	int type;
	bool initialized;
	DRShaderGI * shader;
	float range;                // maximum light gathering range for GI
	int samples;                // GI-method-dependent samples count for global illumination 
	int bounces;                // number of GI bounces. Default 1
	float factor;               // GI contribution 
	set<int> gi_lights;         // IDs of light sources that take part in GI calculations (see also vbo and world node parsing)
	char * param_string;        // Extra parameters that are specific to each renderer.

public:
	GlobalIlluminationRenderer();
	~GlobalIlluminationRenderer();
	virtual bool init(class DeferredRenderer * renderer);
	
	void  addGILight(int ID) {gi_lights.insert(ID);}
	void  removeGILight(int ID) {gi_lights.erase(ID);}
	int   getNumGILights() {return gi_lights.size();}

	virtual void update();
	virtual void draw();
	
	void  setRange(float r) {range = r>0.0f?r:range;}
	float getRange() {return range;}
	
	void  setNumSamples(int s) {samples = s>1?s:1;}
	int   getNumSamples() {return samples;}
	
	void  setFactor(float f) {factor = f>0.0f?f:0.0f;}
	float getFactor() {return factor;}

	void setBounces(int b) {bounces = b>-1?b:0;}
	int  getBounces() {return bounces;}

	bool  isInitialized() {return initialized;}
	
	int   getType() {return type;}

	void  setParamString(char * s) {if (param_string) free(param_string); param_string = strdup(s);}
	char *getParamString() {return param_string;}
};

#endif