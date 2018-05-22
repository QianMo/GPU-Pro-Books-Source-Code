#ifndef _DR_SHADER_MRT_
#define _DR_SHADER_MRT_


#include "DeferredRendererShader.h"

class DRShaderMRT : public DRShader
{
private:
	GLint uniform_texture1,
		  uniform_texture2,
		  uniform_noise,
		  uniform_bump,
		  uniform_ewa,
		  uniform_specular,
		  uniform_emission,
		  uniform_time,
		  uniform_ambient;

	static char DRSH_Vertex[],
		        DRSH_Fragment_Header[],
				DRSH_Fragment_Footer[],
				DRSH_Fragment_Color[],
				DRSH_Fragment_Normal[],
				DRSH_Fragment_Specular[],
				DRSH_Fragment_Lighting[];
	bool initialized;
	double time;

public:
	DRShaderMRT();
	virtual ~DRShaderMRT();
	virtual void start();
	virtual bool init(class DeferredRenderer* _renderer);
	void setTime(double t) {time=t;}
};


#endif