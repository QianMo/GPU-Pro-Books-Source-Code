#include "shaders/DeferredRendererShader_ViewPhotonMap.h"
#include "DeferredRenderer.h"
#include "SceneGraph.h"
#include "GlobalIlluminationRenderer_IV.h"

DRShaderViewPhotonMap::DRShaderViewPhotonMap()
{
	initialized = false;
}

void DRShaderViewPhotonMap::start()
{
	DRShader::start();

	GlobalIlluminationRendererIV * gi_iv = dynamic_cast<GlobalIlluminationRendererIV*>(renderer->getGIRenderer());

	glEnable(GL_TEXTURE_3D); glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_3D, gi_iv->getSHRBuffer());
	glEnable(GL_TEXTURE_3D); glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_3D, gi_iv->getSHGBuffer());
	glEnable(GL_TEXTURE_3D); glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_3D, gi_iv->getSHBBuffer());

	shader->setUniform1i(0,0,uniform_viewphotonmap_shR_buffer);
	shader->setUniform1i(0,1,uniform_viewphotonmap_shG_buffer);
	shader->setUniform1i(0,2,uniform_viewphotonmap_shB_buffer);
	shader->setUniform1i(0,1,uniform_viewphotonmap_buffer_negative);
}

void DRShaderViewPhotonMap::stop()
{
	// disable here all texture units used
	glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_3D, 0); glDisable(GL_TEXTURE_3D);
	glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_3D, 0); glDisable(GL_TEXTURE_3D);
	glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_3D, 0); glDisable(GL_TEXTURE_3D);

	DRShader::stop();
}

bool DRShaderViewPhotonMap::init(class DeferredRenderer* _renderer)
{
	if (_renderer->getGIRenderer() == NULL)
		return false;

	if (! _renderer->getGIRenderer()->isInitialized())
		return false;

	GlobalIlluminationRendererIV * gi_iv = dynamic_cast<GlobalIlluminationRendererIV*>(_renderer->getGIRenderer());
	if (gi_iv == NULL)
		return false;

	if (initialized)
		return true;
	
	if (!DRShader::init(_renderer))
		return false;

	char * shader_text_vert = DRSH_ViewPhotonMap_Vertex;
	char * shader_text_frag = DRSH_ViewPhotonMap_Fragment;

	shader = shader_manager.loadfromMemory ("Common Vertex", shader_text_vert, "View Photon Map", shader_text_frag);
	if (!shader)
	{
		EAZD_TRACE ("DeferredRenderer::buildShaders() : ERROR - Problem compiling view photon map shader.");
		return false;
	}
	else
	{
	    uniform_viewphotonmap_shR_buffer = shader->GetUniformLocation("shR_buffer");
	    uniform_viewphotonmap_shG_buffer = shader->GetUniformLocation("shG_buffer");
	    uniform_viewphotonmap_shB_buffer = shader->GetUniformLocation("shB_buffer");
	    uniform_viewphotonmap_buffer_negative = shader->GetUniformLocation("buffer_negative");
	}
	
	initialized = true;
	return true;
}

//----------------- Shader text ----------------------------

char DRShaderViewPhotonMap::DRSH_ViewPhotonMap_Vertex[] = "\n\
\n\
#version 330 compatibility \n\
\n\
void main (void) \n\
{ \n\
	gl_Position = ftransform (); \n\
    gl_TexCoord[0] = gl_MultiTexCoord0; \n\
}";

char DRShaderViewPhotonMap::DRSH_ViewPhotonMap_Fragment[] = "\n\
\n\
#version 330 compatibility \n\
\n\
uniform sampler3D   shR_buffer, shG_buffer, shB_buffer; \n\
uniform int         buffer_negative; \n\
\n\
// Map [-1.0,1.0] to [0.0,1.0] \n\
#define MAP_MINUS1TO1_0TO1(_value)    (0.5 * (_value) + 0.5) \n\
\n\
void  \n\
main (void) \n\
{ \n\
    // get the value of the current cell \n\
    vec4    colorR = texture3D (shR_buffer, gl_TexCoord[0].xyz); \n\
    vec4    colorG = texture3D (shG_buffer, gl_TexCoord[0].xyz); \n\
    vec4    colorB = texture3D (shB_buffer, gl_TexCoord[0].xyz); \n\
	vec4	pixel  = 0.33 * (colorR + colorG + colorB); \n\
    if (all (equal (pixel.rgb, vec3 (0.0, 0.0, 0.0)))) \n\
        discard; \n\
\n\
    gl_FragColor = buffer_negative * MAP_MINUS1TO1_0TO1 (pixel) + (1-buffer_negative) * pixel; \n\
}";
