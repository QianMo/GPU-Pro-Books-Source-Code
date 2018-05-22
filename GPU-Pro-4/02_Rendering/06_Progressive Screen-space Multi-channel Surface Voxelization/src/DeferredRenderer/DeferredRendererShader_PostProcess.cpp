#include "shaders/DeferredRendererShader_PostProcess.h"
#include "shaders/DeferredRendererShader_FilterKernels.h"
#include "DeferredRenderer.h"

DRShaderPost::DRShaderPost()
{
	hdr_method = -1;
	initialized = false;
}

void DRShaderPost::start()
{
	DRShader::start();
	
	shader->setUniform1i(0,6,uniform_postprocess_framebuffer);
	shader->setUniform1i(0,7,uniform_postprocess_RT_lighting);
	shader->setUniform1i(0,8,uniform_postprocess_depth);
	shader->setUniform1i(0,renderer->getWidth(),uniform_postprocess_width);
	shader->setUniform1i(0,renderer->getHeight(),uniform_postprocess_height);
	shader->setUniform1f(0,renderer->getHDRKey(),uniform_postprocess_hdr_key);
	Vector3D wp = renderer->getHDRWhitePoint();
	shader->setUniform3f(0,wp.x,wp.y,wp.z,uniform_postprocess_hdr_white);
	shader->setUniformMatrix4fv(0,1,false,fmat_P_inv,uniform_postprocess_P_inv);
}

bool DRShaderPost::init(class DeferredRenderer* _renderer)
{
	renderer = _renderer;

	int new_hdr_method = renderer->getHDRMethod();

	if (new_hdr_method!=hdr_method)
		initialized = false;

	if (initialized)
		return true;

	if (!DRShader::init(_renderer))
		return false;

	hdr_method = new_hdr_method;

	char * shader_text_vert;
    char * shader_text_frag;

	shader_text_vert = DRSH_Vertex;
	shader_text_frag = (char*)malloc(20000);
	memset(shader_text_frag, 0, 20000);

	strcpy(shader_text_frag,DRSH_Postprocess_Fragment_Header);
	
	if (hdr_method==DR_HDR_AUTO) 
		strcat(shader_text_frag,DRSH_Postprocess_ToneMapping_Auto);
	else
		strcat(shader_text_frag,DRSH_Postprocess_ToneMapping_Manual);
	
	strcat(shader_text_frag,DRShader_Kernels::DRSH_Gauss_Samples);
	strcat(shader_text_frag,DRSH_Postprocess_Fragment_Color);
	strcat(shader_text_frag,DRSH_Postprocess_Fragment_ToneMapping);
	strcat(shader_text_frag,DRSH_Postprocess_Fragment_Footer);

    shader = shader_manager.loadfromMemory ("Common Vertex", shader_text_vert, "Post-Process", shader_text_frag);
	if (!shader)
	{
		EAZD_TRACE ("DeferredRenderer::buildShaders() : ERROR - Problem compiling postprocess shaders.");
		return false;
	}
	else
	{
        uniform_postprocess_framebuffer = shader->GetUniformLocation("framebuffer");
		uniform_postprocess_height = shader->GetUniformLocation("height");
		uniform_postprocess_width = shader->GetUniformLocation("width");
		uniform_postprocess_RT_lighting = shader->GetUniformLocation("RT_lighting");
		uniform_postprocess_depth = shader->GetUniformLocation("depth");
		uniform_postprocess_P_inv = shader->GetUniformLocation("P_inv");
		uniform_postprocess_hdr_key = shader->GetUniformLocation("hdr_key");
		uniform_postprocess_hdr_white = shader->GetUniformLocation("hdr_white");
	}

	free(shader_text_frag);
	initialized = true;
	return true;
}


//----------------- Shader text ----------------------------

char DRShaderPost::DRSH_Vertex[] = "\n\
varying vec3 Necs; \n\
varying vec4 Pecs; \n\
varying vec3 Tecs; \n\
varying vec3 Becs; \n\
attribute vec3 tangent; \n\
void main(void) \n\
{ \n\
   gl_Position = ftransform(); \n\
   Necs = normalize ( gl_NormalMatrix * gl_Normal ); \n\
   Tecs = normalize ( gl_NormalMatrix * tangent ); \n\
   Becs = cross(Necs,Tecs); \n\
   Pecs = gl_ModelViewMatrix * gl_Vertex; \n\
   gl_TexCoord[0] = gl_TextureMatrix[0]*gl_MultiTexCoord0; \n\
   gl_TexCoord[1] = gl_TextureMatrix[1]*gl_MultiTexCoord1; \n\
   gl_TexCoord[2] = gl_TextureMatrix[2]*gl_MultiTexCoord2; \n\
}";

char DRShaderPost::DRSH_Postprocess_Fragment_Header[] = "\n\
varying vec3  Necs; \n\
varying vec4  Pecs; \n\
uniform int   width; \n\
uniform int   height; \n\
uniform mat4  P_inv; \n\
uniform sampler2D depth; \n\
uniform sampler2D framebuffer; \n\
uniform sampler2D RT_lighting; \n\
uniform vec3  hdr_white; \n\
uniform float hdr_key; \n\
\n\
";

char DRShaderPost::DRSH_Postprocess_ToneMapping_Manual[] = "\n\
vec4 toneMapping(in vec4 col, in vec3 white, in float key) \n\
{ \n\
    vec4 result = vec4(0.0,0.0,0.0,col.a); \n\
	result.r = col.r*(1.0+key*col.r/(white.r*white.r))/(1.0+key*col.r); \n\
	result.g = col.g*(1.0+key*col.g/(white.g*white.g))/(1.0+key*col.g); \n\
	result.b = col.b*(1.0+key*col.b/(white.b*white.b))/(1.0+key*col.b); \n\
	return result; \n\
} \n\
";
char DRShaderPost::DRSH_Postprocess_ToneMapping_Auto[] = "\n\
vec4 toneMapping(in vec4 col, in vec3 white, in float key) \n\
{ \n\
    vec4 result = vec4(0.0,0.0,0.0,col.a); \n\
	vec4 ave = 4.0*texture2DLod(framebuffer, vec2(0.5,0.5),11.0); \n\
	vec3 col_norm = col.rgb*key/(0.1+0.9*(ave.x+ave.y+ave.z)/3.0); \n\
	result.r = col_norm.r*(1.0+col_norm.r/(white.r*white.r))/(1.0+col_norm.r); \n\
	result.g = col_norm.g*(1.0+col_norm.g/(white.g*white.g))/(1.0+col_norm.g); \n\
	result.b = col_norm.b*(1.0+col_norm.b/(white.b*white.b))/(1.0+col_norm.b); \n\
	return result; \n\
} \n\
";

char DRShaderPost::DRSH_Postprocess_Fragment_Color[] = "\n\
void main(void) \n\
{ \n\
initGaussKernel (); \n\
int i; \n\
vec2 offset; \n\
offset = vec2(0.7/width, 0.7/height); \n\
vec4 color = vec4(4,4,4,1)*texture2D(framebuffer,gl_TexCoord[0]); \n\
float total_weight = 0; \n\
for (i=0; i<16; i++) {\n\
float weight = exp(-2*dot(vec2(kernel[2*i],kernel[2*i+1]),vec2(kernel[2*i],kernel[2*i+1]))); \n\
color+=weight*vec4(4,4,4,1)*texture2D(framebuffer,gl_TexCoord[0]+vec2(offset.x*kernel[2*i],offset.y*kernel[2*i+1])); \n\
total_weight+=weight; \n\
}\n\
color/=total_weight; \n\
";

char DRShaderPost::DRSH_Postprocess_Fragment_ToneMapping[] = "\n\
color=toneMapping(color,hdr_white,hdr_key);";

char DRShaderPost::DRSH_Postprocess_Fragment_Footer[] = "\n\
gl_FragColor = vec4(color.rgb,1.0); \n\
}";