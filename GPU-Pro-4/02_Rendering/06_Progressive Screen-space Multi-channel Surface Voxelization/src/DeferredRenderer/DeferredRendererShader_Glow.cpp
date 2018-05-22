#include "shaders/DeferredRendererShader_Glow.h"
#include "shaders/DeferredRendererShader_PostProcess.h"
#include "DeferredRenderer.h"

DRShaderGlow::DRShaderGlow()
{
	initialized = false;
	hdr_method = -1;
}

void DRShaderGlow::start()
{
	DRShader::start();
	
	shader->setUniform1i(0,5,uniform_glow_RT_lighting);
	shader->setUniform1i(0,renderer->getWidth(),uniform_glow_width);
	shader->setUniform1i(0,renderer->getHeight(),uniform_glow_height);
	shader->setUniform1f(0,renderer->getHDRKey(),uniform_glow_hdr_key);
	Vector3D wp = renderer->getHDRWhitePoint();
	shader->setUniform3f(0,wp.x,wp.y,wp.z,uniform_glow_hdr_white);
}

bool DRShaderGlow::init(class DeferredRenderer* _renderer)
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

	strcpy(shader_text_frag,DRSH_Glow_Fragment_Header);
	if (hdr_method==DR_HDR_AUTO) 
		strcat(shader_text_frag,DRShaderPost::DRSH_Postprocess_ToneMapping_Auto);
	else
		strcat(shader_text_frag,DRShaderPost::DRSH_Postprocess_ToneMapping_Manual);
	strcat(shader_text_frag,DRSH_Glow_Fragment_Core);
	strcat(shader_text_frag,DRSH_Glow_Fragment_Footer);

    shader = shader_manager.loadfromMemory ("Common Vertex", shader_text_vert, "Glow", shader_text_frag);
	if (!shader)
	{
		EAZD_TRACE ("DeferredRenderer::buildShaders() : ERROR - Problem compiling glow shaders.");
		return false;
	}
	else
	{
        uniform_glow_height = shader->GetUniformLocation("height");
		uniform_glow_width =  shader->GetUniformLocation("width");
		uniform_glow_RT_lighting = shader->GetUniformLocation("framebuffer");
		uniform_glow_hdr_key = shader->GetUniformLocation("hdr_key");
		uniform_glow_hdr_white = shader->GetUniformLocation("hdr_white");
	}

	free(shader_text_frag);
	initialized = true;
	return true;
}


//----------------- Shader text ----------------------------

char DRShaderGlow::DRSH_Vertex[] = "\n\
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

char DRShaderGlow::DRSH_Glow_Fragment_Header[] = "\n\
varying vec3  Necs; \n\
varying vec4  Pecs; \n\
uniform int   width; \n\
uniform int   height; \n\
uniform vec3  hdr_white; \n\
uniform float hdr_key; \n\
uniform sampler2D framebuffer;";

char DRShaderGlow::DRSH_Glow_Fragment_Core[] = "\n\
void main(void) \n\
{ \n\
int i,j; \n\
vec2 offset = vec2(8.0/width, 8.0/height); \n\
vec4 glow = vec4(0,0,0,1); //2*texture2D(framebuffer,gl_TexCoord[0]); \n\
//glow.rgb+=clamp(glow.rgb-vec3(1.0,1.0,1.0),vec3(0,0,0),vec3(1,1,1)); \n\
for (i=-2; i<2; i++) {\n\
for (j=-2; j<2; j++) \n\
{ \n\
    vec2 tc = vec2(i*offset.x,j*offset.y); \n\
	vec4 tone = toneMapping(vec4(4,4,4,1)*texture2D(framebuffer,gl_TexCoord[0]+tc),hdr_white,hdr_key); \n\
	glow.rgb+=clamp(tone.rgb-vec3(1.0,1.0,1.0),vec3(0,0,0),vec3(1,1,1)); \n\
	}} \n\
glow.rgb/=25.0; \n\
";

char DRShaderGlow::DRSH_Glow_Fragment_Footer[] = "\n\
gl_FragColor = glow; \n\
}";