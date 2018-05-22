#include "shaders/DeferredRendererShader_ClearMRT.h"
#include "DeferredRenderer.h"

DRShaderClearMRT::DRShaderClearMRT()
{
	initialized = false;
}

void DRShaderClearMRT::start()
{
	DRShader::start();

	float background[3];
	renderer->getBackground(background, background+1, background+2);
	shader->setUniform3f(0,background[0],background[1],background[2],uniform_clear_background);
}

bool DRShaderClearMRT::init(class DeferredRenderer* _renderer)
{
	if (initialized)
		return true;
	
	if (!DRShader::init(_renderer))
		return false;

	char * shader_text_vert;
    char * shader_text_frag;

	shader_text_vert = DRSH_Vertex;
	shader_text_frag = (char*)malloc(20000);
	memset(shader_text_frag, 0, 20000);
	strcpy(shader_text_frag,DRSH_Fragment_Clear);

	shader = shader_manager.loadfromMemory ("Common Vertex", shader_text_vert, "MRT Clear", shader_text_frag);
	if (!shader)
	{
		EAZD_TRACE ("DeferredRenderer::buildShaders() : ERROR - Problem compiling clear shader.");
		free(shader_text_frag);
		return false;
	}
	else
	{
	    uniform_clear_background = shader->GetUniformLocation("background");
	}
	
	free(shader_text_frag);
	initialized = true;
	return true;

}

//----------------- Shader text ----------------------------

char DRShaderClearMRT::DRSH_Fragment_Clear[] = "\n\
varying vec3 Necs; \n\
varying vec4 Pecs; \n\
uniform vec3 background; \n\
void main(void) \n\
{ \n\
gl_FragData[0] = vec4(background,1.0); \n\
gl_FragData[1] = vec4(0,0,0,1); \n\
gl_FragData[2] = vec4(0,0,0,0); \n\
gl_FragData[3] = vec4(0,0,0,0); \n\
}";

char DRShaderClearMRT::DRSH_Vertex[] = "\n\
varying vec3 Necs; \n\
varying vec4 Pecs; \n\
varying vec3 Tecs; \n\
varying vec3 Becs; \n\
attribute vec3 tangent; \n\
void main(void) \n\
{ \n\
   //gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex; \n\
   gl_Position = ftransform(); \n\
   Necs = normalize ( gl_NormalMatrix * gl_Normal ); \n\
   Tecs = normalize ( gl_NormalMatrix * tangent ); \n\
   Becs = cross(Necs,Tecs); \n\
   //Necs = vec4(normalize((gl_ModelViewProjectionMatrix * vec4(gl_Normal, 0.0)).xyz),1); \n\
   Pecs = gl_ModelViewMatrix * gl_Vertex; \n\
   gl_TexCoord[0] = gl_TextureMatrix[0]*gl_MultiTexCoord0; \n\
   gl_TexCoord[1] = gl_TextureMatrix[1]*gl_MultiTexCoord1; \n\
   gl_TexCoord[2] = gl_TextureMatrix[2]*gl_MultiTexCoord2; \n\
}";