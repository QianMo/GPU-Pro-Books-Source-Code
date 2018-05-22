#include "shaders/DeferredRendererShader_Illumination.h"
#include "shaders/DeferredRendererShader_FilterKernels.h"
#include "shaders/DeferredRendererShader_ShadowFunctions.h"
#include "DeferredRenderer.h"
#include <math.h>

void DRShaderIllumination::start()
{
	DRShader::start();
	
	if (!L)
		return;

	float *col = L->getColor();
	Vector3D light_pos = Vector3D(L->getTransformedPosition());
	Vector3D light_dir = Vector3D(L->getTransformedTarget());
	light_dir = light_dir - light_pos;
	light_dir.normalize();

	shader->setUniform3f(0,col[0], col[1], col[2],uniform_illumination_light_col);
	shader->setUniform1i(0,L->isShadowEnabled()?1:0,uniform_illumination_use_shadow);
	shader->setUniform1i(0,renderer->getActualWidth(),uniform_illumination_width);
	shader->setUniform1i(0,renderer->getActualHeight(),uniform_illumination_height);
	shader->setUniform3f(0,light_pos[0], light_pos[1], light_pos[2], uniform_illumination_light_pos);
	shader->setUniform3f(0,light_dir[0], light_dir[1], light_dir[2], uniform_illumination_light_dir);
	shader->setUniform3f(0,eye_pos[0],eye_pos[1],eye_pos[2], uniform_illumination_eye_pos);
	shader->setUniform1i(0,L->isAttenuated(), uniform_illumination_light_attn);
	shader->setUniform1i(0,L->isActive(), uniform_illumination_light_active);
	shader->setUniform1f(0,L->getTransformedFar(), uniform_illumination_light_range);
	shader->setUniform1i(0,L->isConical()?1:0, uniform_illumination_is_cone);
	shader->setUniform1f(0,L->isConical()?cos(0.01745f*L->getApperture()):0.0f, uniform_illumination_cone);
	shader->setUniform1f(0,L->getSize(), uniform_illumination_light_size);
	shader->setUniform1i(0,L->getShadowMapRes(),uniform_illumination_shadow_size);
	shader->setUniform1i(0,6,uniform_illumination_noise);
	shader->setUniform1i(0,7,uniform_illumination_RT_normals);
	shader->setUniform1i(0,8,uniform_illumination_RT_depth);
	shader->setUniform1i(0,9,uniform_illumination_RT_specular);
	shader->setUniform1i(0,0,uniform_illumination_RT_shadow);
	shader->setUniformMatrix4fv(0,1,false,fmat_MVP_inv,uniform_illumination_MVP_inverse);
	shader->setUniformMatrix4fv(0,1,false,fmat_L,uniform_illumination_M_light);
	shader->setUniformMatrix4fv(0,1,false,fmat_P,uniform_illumination_Projection);
}

bool DRShaderIllumination::init(class DeferredRenderer* _renderer)
{
	if (!DRShader::init(_renderer))
		return false;

	char * shader_text_vert;
    char * shader_text_frag;

	shader_text_vert = DRSH_Vertex;
	shader_text_frag = (char*)malloc(20000);
	memset(shader_text_frag, 0, 20000);

	strcpy(shader_text_frag,DRSH_Illumination_Fragment_Header);
	strcat(shader_text_frag,DRShader_Kernels::DRSH_Gauss_Samples);
	strcat(shader_text_frag,DRShader_ShadowFunctions::DRSH_Shadow_Gaussian);
	
	strcat(shader_text_frag,DRSH_Illumination_Fragment_Core);
	strcat(shader_text_frag,DRSH_Illumination_Fragment_Footer);
	
    shader = shader_manager.loadfromMemory ("Common Vertex", shader_text_vert, "Illumination", shader_text_frag);
	if (!shader)
	{
		EAZD_TRACE ("DeferredRenderer::buildShaders() : ERROR - Problem compiling illumination shaders.");
		free(shader_text_frag);
		return false;
	}
	else
	{
        uniform_illumination_width = shader->GetUniformLocation("width");
		uniform_illumination_height = shader->GetUniformLocation("height");
		uniform_illumination_light_pos = shader->GetUniformLocation("light_pos");
		uniform_illumination_eye_pos = shader->GetUniformLocation("eye_pos");
		uniform_illumination_light_dir = shader->GetUniformLocation("light_dir");
		uniform_illumination_light_col = shader->GetUniformLocation("light_col");
		uniform_illumination_light_attn = shader->GetUniformLocation("light_attn");
		uniform_illumination_light_active = shader->GetUniformLocation("light_active");
   		uniform_illumination_light_range = shader->GetUniformLocation("light_range");
   		uniform_illumination_RT_normals = shader->GetUniformLocation("RT_normals");
		uniform_illumination_RT_depth = shader->GetUniformLocation("RT_depth");
		uniform_illumination_RT_specular = shader->GetUniformLocation("RT_specular");
		uniform_illumination_MVP_inverse = shader->GetUniformLocation("MVP_inverse");
		uniform_illumination_Projection = shader->GetUniformLocation("Projection");
		uniform_illumination_M_light = shader->GetUniformLocation("M_light");
		uniform_illumination_light_size = shader->GetUniformLocation("light_size");
		uniform_illumination_use_shadow = shader->GetUniformLocation("use_shadow");
		uniform_illumination_shadow_size = shader->GetUniformLocation("shadow_size");
		uniform_illumination_RT_shadow = shader->GetUniformLocation("RT_shadow");
		uniform_illumination_noise = shader->GetUniformLocation("noise");
		uniform_illumination_cone = shader->GetUniformLocation("cone");
		uniform_illumination_is_cone = shader->GetUniformLocation("is_cone");
	}
	
	free(shader_text_frag);
	return true;
}




//----------------- Shader text ----------------------------


char DRShaderIllumination::DRSH_Vertex[] = "\n\
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

char DRShaderIllumination::DRSH_Illumination_Fragment_Header[] = "\n\
varying vec3  Necs; \n\
varying vec4  Pecs; \n\
uniform int   width; \n\
uniform int   height; \n\
uniform int   shadow_size; \n\
uniform sampler2D noise; \n\
uniform sampler2D RT_normals; \n\
uniform sampler2D RT_depth; \n\
uniform sampler2D RT_specular; \n\
uniform sampler2D RT_shadow; \n\
uniform int   light_attn; \n\
uniform int   use_shadow; \n\
uniform int   light_active; \n\
uniform float light_range; \n\
uniform float light_size; \n\
uniform vec3  light_pos; \n\
uniform vec3  light_dir; \n\
uniform mat4  MVP_inverse; \n\
uniform mat4  Projection; \n\
uniform mat4  M_light; \n\
uniform vec3  eye_pos; \n\
uniform vec3  light_col; \n\
uniform int   is_cone; \n\
uniform float cone; \n\
\n\
vec3 VectorECS2WCS(in vec3 sample) \n\
{ \n\
vec4 vector_WCS = MVP_inverse*Projection*vec4(sample,1); \n\
vec4 zero_WCS = MVP_inverse*Projection*vec4(0,0,0,1); \n\
vector_WCS=vector_WCS/vector_WCS.w-zero_WCS/zero_WCS.w; \n\
return vector_WCS.xyz; \n\
} \n";

char DRShaderIllumination::DRSH_Illumination_Fragment_Footer[] = "\n\
}";

char DRShaderIllumination::DRSH_Illumination_Fragment_Core[] = "\n\
void main(void) \n\
{ \n\
initGaussKernel (); \n\
vec2 screen_coords = vec2(gl_FragCoord.x/width, gl_FragCoord.y/height); \n\
float depth = texture2D(RT_depth,screen_coords).r;\n\
if (depth==1.0) \n\
	discard; \n\
vec4 normalbuf = texture2D(RT_normals,screen_coords);\n\
vec3 normal = normalbuf.xyz;\n\
normal.x = normal.x*2-1;\n\
normal.y = normal.y*2-1;\n\
//float emission = normalbuf.w;\n\
int i; \n\
vec3 pos_normalized = vec3(2*screen_coords.x-1, 2*screen_coords.y-1, 2*depth-1);\n\
vec4 pos_WCS = MVP_inverse*vec4(pos_normalized,1); \n\
vec4 normal_WCS = vec4(normalize(VectorECS2WCS(normal)),1);\n\
vec3 ldir; \n\
float ldist; \n\
ldir = light_pos - pos_WCS.xyz/pos_WCS.w; \n\
ldist = length(ldir); \n\
ldir = normalize(ldir); \n\
float dst_attn = 1.0-light_attn * clamp(ldist,0.0,light_range)/light_range;\n\
vec3 diffuse = dst_attn*light_col*light_active*clamp(dot(ldir,normal_WCS.xyz),0.0,1.0); \n\
vec3 eye_dir = normalize(eye_pos - pos_WCS); \n\
vec3 halfway = normalize(ldir+eye_dir); \n\
vec4 spec_coefs = texture2D(RT_specular,screen_coords); \n\
vec3 specular = vec3(0.0,0.0,0.0); \n\
if (dot(normal_WCS.xyz,ldir)>0) \n\
    specular = spec_coefs.xyz*dst_attn*light_col*pow(clamp(dot(halfway,normal_WCS.xyz),0.0,1.0),spec_coefs.w*127.0); \n\
float cf; \n\
if (is_cone) \n\
{ \n\
    cf = clamp(-40.0*(dot(light_dir,ldir)+cone),0.0,1.0); \n\
	specular*=cf; \n\
	diffuse*=cf; \n\
} \n\
if (!use_shadow) \n\
    gl_FragColor = vec4((diffuse+specular)*0.5,1.0);\n\
else \n\
	gl_FragColor = vec4((diffuse+specular)*0.5,1.0) * \n\
       shadowAttenuation(RT_shadow, pos_WCS/pos_WCS.w, M_light); \n\
";