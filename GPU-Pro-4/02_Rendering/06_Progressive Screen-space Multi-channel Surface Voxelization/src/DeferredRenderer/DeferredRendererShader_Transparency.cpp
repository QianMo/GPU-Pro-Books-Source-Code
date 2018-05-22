#include "shaders/DeferredRendererShader_Transparency.h"
#include "shaders/DeferredRendererShader_FilterKernels.h"
#include "shaders/DeferredRendererShader_ShadowFunctions.h"
#include "DeferredRenderer.h"

DRShaderTransparency::DRShaderTransparency()
{
	initialized = false;
	shadow_method = -1;
}

void DRShaderTransparency::start()
{
	DRShader::start();

	float *col = L->getColor();
	Vector3D light_pos = Vector3D(L->getTransformedPosition());
	Vector3D light_dir = Vector3D(L->getTransformedTarget());
	light_dir = light_dir - light_pos;
	light_dir.normalize();
	float global_ambient[3]; 
	renderer->getAmbient(global_ambient, global_ambient+1, global_ambient+2);

	shader->setUniform1i(0,L->isShadowEnabled()?1:0,uniform_transparency_use_shadow);
	shader->setUniform1i(0,renderer->getActualWidth(),uniform_transparency_width);
	shader->setUniform1i(0,renderer->getActualHeight(),uniform_transparency_height);
	shader->setUniform3f(0,light_pos[0], light_pos[1], light_pos[2], uniform_transparency_light_pos);
	shader->setUniform3f(0,light_dir[0], light_dir[1], light_dir[2], uniform_transparency_light_dir);
	shader->setUniform3f(0,col[0], col[1], col[2],uniform_transparency_light_col);
	shader->setUniform3f(0,eye_pos[0],eye_pos[1],eye_pos[2], uniform_transparency_eye_pos);
	shader->setUniform3f(0,global_ambient[0],global_ambient[1],global_ambient[2], uniform_transparency_ambient_term);
	shader->setUniform1i(0,L->isAttenuated()?1:0, uniform_transparency_light_attn);
	shader->setUniform1i(0,L->isActive()?1:0, uniform_transparency_light_active);
	shader->setUniform1f(0,L->getTransformedFar(), uniform_transparency_light_range);
	shader->setUniform1f(0,L->getSize(), uniform_transparency_light_size);
	shader->setUniform1i(0,0,uniform_transparency_texture1);
	shader->setUniform1i(0,1,uniform_transparency_texture2);
	shader->setUniform1i(0,8,uniform_transparency_RT_depth);
	shader->setUniform1i(0,7,uniform_transparency_RT_shadow);
	shader->setUniform1i(0,9,uniform_transparency_RT_specular);
	shader->setUniform1i(0,renderer->getNumLights(),uniform_transparency_num_lights);
	shader->setUniformMatrix4fv(0,1,false,fmat_MVP_inv,uniform_transparency_MVP_inverse);
	shader->setUniformMatrix4fv(0,1,false,fmat_L,uniform_transparency_M_light);
}

bool DRShaderTransparency::init(DeferredRenderer* _renderer)
{
	
	renderer = _renderer;
	int new_shadow_method = renderer->getShadowMethod();
	if (new_shadow_method!=shadow_method)
		initialized = false;

	if (initialized)
		return true;

	if (!DRShader::init(_renderer))
		return false;

	shadow_method = new_shadow_method;

	char * shader_text_vert;
    char * shader_text_frag;

	shader_text_vert = DRSH_Transparency_Vertex;
	shader_text_frag = (char*)malloc(20000);
	memset(shader_text_frag, 0, 20000);

	strcpy(shader_text_frag,DRSH_Transparency_Fragment_Header);
	strcat(shader_text_frag,DRShader_Kernels::DRSH_Gauss_Samples);
	strcat(shader_text_frag,DRShader_ShadowFunctions::DRSH_Shadow_Gaussian);
	strcat(shader_text_frag,DRSH_Transparency_Fragment_Core);

    shader = shader_manager.loadfromMemory ("Transparency Vert", DRSH_Transparency_Vertex, 
	                                        "Transparency Frag", shader_text_frag);
	if (!shader)
	{
		EAZD_TRACE ("DeferredRenderer::buildShaders() : ERROR - Problem compiling transparency shaders.");
		return false;
	}
	else
	{
        uniform_transparency_width = shader->GetUniformLocation("width");
		uniform_transparency_height = shader->GetUniformLocation("height");
		uniform_transparency_eye_pos = shader->GetUniformLocation("eye_pos");
		uniform_transparency_light_dir = shader->GetUniformLocation("light_dir");
		uniform_transparency_light_pos = shader->GetUniformLocation("light_pos");
		uniform_transparency_light_col = shader->GetUniformLocation("light_col");
		uniform_transparency_light_attn = shader->GetUniformLocation("light_attn");
		uniform_transparency_light_active = shader->GetUniformLocation("light_active");
		uniform_transparency_light_range = shader->GetUniformLocation("light_range");
		uniform_transparency_MVP_inverse = shader->GetUniformLocation("MVP_inverse");
		uniform_transparency_light_size = shader->GetUniformLocation("light_size");
		uniform_transparency_M_light = shader->GetUniformLocation("M_light");
		uniform_transparency_use_shadow = shader->GetUniformLocation("use_shadow");
		uniform_transparency_RT_shadow = shader->GetUniformLocation("RT_shadow");
		uniform_transparency_RT_depth = shader->GetUniformLocation("RT_depth");
		uniform_transparency_RT_specular = shader->GetUniformLocation("RT_specular");
		uniform_transparency_num_lights = shader->GetUniformLocation("num_lights");
		uniform_transparency_ambient_term = shader->GetUniformLocation("ambient_term");
		uniform_transparency_texture1 = shader->GetUniformLocation("texture1");
		uniform_transparency_texture2 = shader->GetUniformLocation("texture2");
	}

	free(shader_text_frag);
	initialized = true;
	return true;
}


//----------------- Shader text ----------------------------

char DRShaderTransparency::DRSH_Transparency_Vertex[] = "\n\
varying vec4 Pwcs; \n\
varying vec4 Necs; \n\
uniform mat4 MVP_inverse; \n\
void main(void) \n\
{ \n\
   gl_Position = ftransform(); \n\
   Pwcs = MVP_inverse * gl_Position; \n\
   Pwcs/=Pwcs.w; \n\
   vec4 Zero = MVP_inverse * gl_ModelViewProjectionMatrix * vec4(0,0,0,1); \n\
   //Necs = vec4(normalize ( gl_NormalMatrix* gl_Normal ),1); \n\
   Necs = vec4(normalize((gl_ModelViewProjectionMatrix * vec4(gl_Normal, 0.0)).xyz),1); \n\
   gl_TexCoord[0] = gl_TextureMatrix[0]*gl_MultiTexCoord0; \n\
   gl_TexCoord[1] = gl_TextureMatrix[1]*gl_MultiTexCoord1; \n\
   gl_TexCoord[2] = gl_TextureMatrix[2]*gl_MultiTexCoord2; \n\
}";


char DRShaderTransparency::DRSH_Transparency_Fragment_Header[] = "\n\
varying vec4  Necs; \n\
varying vec4  Pwcs; \n\
uniform int   width; \n\
uniform int   height; \n\
uniform sampler2D texture1; \n\
uniform sampler2D texture2; \n\
uniform sampler2D RT_shadow; \n\
uniform sampler2D RT_specular; \n\
uniform sampler2D RT_depth; \n\
uniform int   shadow_size; \n\
uniform sampler2D noise; \n\
uniform int   light_attn; \n\
uniform int   use_shadow; \n\
uniform int   light_active; \n\
uniform float light_range; \n\
uniform float light_size; \n\
uniform vec3  light_pos; \n\
uniform vec3  light_dir; \n\
uniform mat4  MVP_inverse; \n\
uniform mat4  M_light; \n\
uniform vec3  eye_pos; \n\
uniform int   num_lights; \n\
uniform vec3 ambient_term; \n\
uniform vec3  light_col; \n";

char DRShaderTransparency::DRSH_Transparency_Fragment_Core[] = "\n\
vec3 VectorCSS2WCS(in vec3 sample) \n\
{ \n\
vec4 vector_WCS = MVP_inverse*vec4(sample,1); \n\
vec4 zero_WCS = MVP_inverse*vec4(0,0,0,1); \n\
vector_WCS=vector_WCS/vector_WCS.w-zero_WCS/zero_WCS.w; \n\
return vector_WCS.xyz; \n\
} \n\
\n\
float detectTexture(in sampler2D tex) \n\
{ \n\
   vec4 test = texture2D(tex, vec2(0.5,0.5))+ \n\
               texture2D(tex, vec2(0.25,0.25)); \n\
   return (sign (test.r+test.g+test.b+test.a-0.0001)+1)/2; \n\
} \n\
void main(void) \n\
{ \n\
initGaussKernel (); \n\
vec2 frag_offset = vec2(gl_FragCoord.x/width, gl_FragCoord.y/height); \n\
float depth = texture2D(RT_depth,frag_offset).r;\n\
if (depth<gl_FragCoord.z) \n\
   discard; \n\
\n\
vec3 Nwcs = normalize(VectorCSS2WCS(Necs.xyz/Necs.w)); \n\
vec4 diffuse = vec4(0,0,0,0); \n\
vec3 ldir; \n\
float ldist; \n\
ldir = light_pos - Pwcs.xyz/Pwcs.w; \n\
ldist = length(ldir); \n\
ldir = normalize(ldir); \n\
float dst_attn = 1.0-light_attn * clamp(ldist,0.0,light_range)/light_range;\n\
diffuse += gl_FrontMaterial.diffuse * dst_attn*vec4(light_col,1)* \n\
           light_active*clamp(dot(ldir,Nwcs),0.0,1.0); \n\
vec3 emission = gl_FrontMaterial.emission.rgb; \n\
vec4 tex_color1 = mix(vec4(1,1,1,1),texture2D(texture1, gl_TexCoord[0].st),detectTexture(texture1)); \n\
vec4 tex_color2 = texture2D(texture2, gl_TexCoord[1].st); \n\
tex_color2.a *= detectTexture(texture2); \n\
vec4 tex_comb = mix(tex_color1,tex_color2,tex_color2.a); \n\
diffuse *= tex_comb; \n\
vec3 eye_dir = normalize(eye_pos - Pwcs); \n\
vec3 halfway = normalize(ldir+eye_dir); \n\
vec4 spec_coefs = vec4(gl_FrontMaterial.specular.xyz,gl_FrontMaterial.shininess); \n\
vec3 specular = vec3(0.0,0.0,0.0); \n\
float highlight = 0; \n\
if (dot(Nwcs,ldir)>0) \n\
{ \n\
highlight = pow(clamp(dot(halfway,Nwcs),0.0,1.0),spec_coefs.w); \n\
specular = spec_coefs.xyz*dst_attn*light_col*highlight; \n\
} \n\
float sh; \n\
if (use_shadow) \n\
	sh = shadowAttenuation(RT_shadow, Pwcs, M_light); \n\
else \n\
	sh = 1.0; \n\
	gl_FragColor = vec4(0.25,0.25,0.25,1.0)*(vec4(ambient_term+diffuse.rgb*sh/2+emission*tex_comb.rgb*tex_comb.a/2,gl_FrontMaterial.diffuse.a*tex_comb.a)+vec4(specular*sh/2,highlight));\n\
}";
