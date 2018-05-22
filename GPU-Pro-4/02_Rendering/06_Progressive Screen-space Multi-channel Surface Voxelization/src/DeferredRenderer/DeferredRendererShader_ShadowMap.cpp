#include "shaders/DeferredRendererShader_ShadowMap.h"
#include "DeferredRenderer.h"

DRShaderShadowMap::DRShaderShadowMap()
{
	initialized = false;
	L = NULL;
}

DRShaderShadowMap::~DRShaderShadowMap()
{

}

void DRShaderShadowMap::start()
{
	DRShader::start();

	shader->setUniform1i(0,0,uniform_texture1);		
	shader->setUniform1i(0,4,uniform_emission);

	if (L)
	{
		float * color = L->getColor();

		shader->setUniform3f(0,color[0], color[1], color[2],uniform_light_color);	
		shader->setUniform1i(0,L->isAttenuated()?1:0,uniform_attenuating);
		float f = L->getTransformedFar();
		shader->setUniform1f(0,f,uniform_far);
		shader->setUniform1i(0,L->isConical()?1:0,uniform_use_cone);
		shader->setUniform1f(0,L->getCone(),uniform_cone);
	}
	else
	{
		shader->setUniform3f(0,1.0f, 1.0f, 1.0f,uniform_light_color);	
		shader->setUniform1i(0,0,uniform_attenuating);
		shader->setUniform1f(0,1.0f,uniform_far);
	}
}

bool DRShaderShadowMap::init(class DeferredRenderer* _renderer)
{
	if (initialized)
		return true;

	if (!DRShader::init(_renderer))
		return false;

	char * shader_text_vert;
    char * shader_text_frag;

	shader_text_vert = DRSH_Vertex;
	shader_text_frag = DRSH_Fragment;

	shader = shader_manager.loadfromMemory ("Common Vertex", shader_text_vert, "Extended shadow map", shader_text_frag);
	if (!shader)
	{
		EAZD_TRACE ("DeferredRenderer::buildShaders() : ERROR - Problem compiling shadow map shader.");
		free(shader_text_frag);
		return false;
	}
	else
	{
		uniform_texture1 = shader->GetUniformLocation("texture1");
		uniform_emission = shader->GetUniformLocation("emission");
		uniform_light_color = shader->GetUniformLocation("light_color");
		uniform_attenuating = shader->GetUniformLocation("attenuating");
		uniform_far = shader->GetUniformLocation("far");
		uniform_cone = shader->GetUniformLocation("cone");
		uniform_use_cone = shader->GetUniformLocation("use_cone");
	}

	initialized = true;
	return true;

}

//----------------- Shader text ----------------------------

char DRShaderShadowMap::DRSH_Vertex[] = "\n\
varying vec3 Necs; \n\
varying vec4 Pecs; \n\
varying vec4 Pcss; \n\
void main(void) \n\
{ \n\
   gl_Position = Pcss = ftransform(); \n\
   Necs = gl_NormalMatrix * gl_Normal; \n\
   Pecs = gl_ModelViewMatrix * gl_Vertex; \n\
   gl_TexCoord[0] = gl_TextureMatrix[0]*gl_MultiTexCoord0; \n\
}";

char DRShaderShadowMap::DRSH_Fragment[] = "\n\
varying vec3 Necs; \n\
varying vec4 Pecs; \n\
varying vec4 Pcss; \n\
uniform sampler2D texture1, emission; \n\
uniform int attenuating; \n\
uniform float cone; \n\
uniform int use_cone; \n\
uniform vec3 light_color; \n\
uniform float far; \n\
float detectTexture(in sampler2D tex) \n\
{ \n\
   vec4 test = texture2D(tex, vec2(0.5,0.5))+ \n\
               texture2D(tex, vec2(0.25,0.25)); \n\
   return (sign (test.r+test.g+test.b+test.a-0.0001)+1)/2; \n\
} \n\
\n\
void main(void) \n\
{\n\
float hastex1 = detectTexture(texture1); \n\
vec4 tex_color = mix(vec4(1,1,1,1),texture2D(texture1, gl_TexCoord[0].st),hastex1); \n\
vec4 tex_comb = vec4(light_color,1.0)*gl_FrontMaterial.diffuse*tex_color; \n\
float alpha_clamp = max(0.0,sign(tex_comb.a-0.5)); \n\
if (alpha_clamp<0.5) discard; \n\
float em = (gl_FrontMaterial.emission.x+gl_FrontMaterial.emission.y+gl_FrontMaterial.emission.z)/3.0; \n\
em += texture2D(emission, gl_TexCoord[0]).r;\n\
vec3 n = normalize(Necs); \n\
float falloff = 1.0-attenuating*clamp(length(Pecs.xyz),0.0,far)/far; \n\
falloff *= use_cone?(1.0-clamp(length(Pcss.xy)/cone-0.5,0.0,1.0)):1.0; \n\
gl_FragData[0] = vec4(falloff*n.z*tex_comb.rgb,alpha_clamp); \n\
gl_FragData[1] = vec4(0.5+n.x*0.5, 0.5+n.y*0.5,n.z, 1-em); \n\
}";

