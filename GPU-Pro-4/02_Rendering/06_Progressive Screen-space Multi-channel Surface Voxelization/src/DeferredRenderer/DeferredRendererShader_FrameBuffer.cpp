#include "shaders/DeferredRendererShader_FrameBuffer.h"
#include "shaders/DeferredRendererShader_FilterKernels.h"
#include "DeferredRenderer.h"

DRShaderFrameBuffer::DRShaderFrameBuffer()
{
	units_per_meter = 1.0f;
}

void DRShaderFrameBuffer::start()
{
	DRShader::start();

	float global_ambient[3];
	renderer->getAmbient(global_ambient, global_ambient+1, global_ambient+2);

	shader->setUniform1i(0,2,uniform_RT_albedo);
	shader->setUniform1i(0,3,uniform_RT_normals);
	shader->setUniform1i(0,4,uniform_RT_specular);
	shader->setUniform1i(0,5,uniform_RT_lighting);
	shader->setUniform1i(0,6,uniform_RT_depth);
	shader->setUniform1i(0,7,uniform_RT_ao);
	shader->setUniform1i(0,8,uniform_noise);
	int blending = renderer->isGIEnabled()?1:renderer->getAmbientBlendMode();
	shader->setUniform1i(0,blending,uniform_use_gi);
	shader->setUniform1i(0,renderer->getWidth(),uniform_width);
	shader->setUniform1i(0,renderer->getHeight(),uniform_height);
	shader->setUniform3f(0,global_ambient[0],
		                   global_ambient[1],
						   global_ambient[2],uniform_ambient_term);
}

bool DRShaderFrameBuffer::init(class DeferredRenderer* _renderer)
{
	if (!DRShader::init(_renderer))
		return false;

	char * shader_text_vert;
    char * shader_text_frag;

	shader_text_vert = DRSH_Vertex;
	shader_text_frag = (char*)malloc(20000);
	memset(shader_text_frag, 0, 20000);
	
	strcpy(shader_text_frag,DRSH_Framebuffer_Fragment_Header);
	strcat(shader_text_frag,DRShader_Kernels::DRSH_Gauss_Samples);
	
	if (renderer->isGIEnabled())
		strcat(shader_text_frag,DRSH_Framebuffer_Fragment_Core);
	else
		strcat(shader_text_frag,DRSH_Framebuffer_Fragment_Core_MultiSampledGI);
	
	strcat(shader_text_frag,DRSH_Framebuffer_Fragment_Footer);
	
    shader = shader_manager.loadfromMemory ("Common Vertex", shader_text_vert, "Composited Framebuffer", shader_text_frag);
	if (!shader)
	{
		EAZD_TRACE ("DeferredRenderer::buildShaders() : ERROR - Problem compiling framebuffer shaders.");
		free(shader_text_frag);
		return false;
	}
	else
	{
        uniform_RT_albedo = shader->GetUniformLocation("RT_albedo");
		uniform_RT_normals = shader->GetUniformLocation("RT_normals");
		uniform_RT_lighting = shader->GetUniformLocation("RT_lighting");
		uniform_RT_specular = shader->GetUniformLocation("RT_specular");
		uniform_RT_depth = shader->GetUniformLocation("RT_depth");		
		uniform_RT_ao = shader->GetUniformLocation("RT_ao");
		uniform_width = shader->GetUniformLocation("width");
		uniform_height = shader->GetUniformLocation("height");
		uniform_ambient_term = shader->GetUniformLocation("ambient_term");
		uniform_use_gi = shader->GetUniformLocation("use_gi");
		uniform_sun_wcs = shader->GetUniformLocation("sun_wcs");
		uniform_camera_wcs = shader->GetUniformLocation("camera_wcs");
		uniform_unitspm = shader->GetUniformLocation("unitspm");
		uniform_MVP_inverse = shader->GetUniformLocation("MVP_inverse");
		uniform_Projection = shader->GetUniformLocation("Projection");
		uniform_noise = shader->GetUniformLocation("noise");
		uniform_shadow = shader->GetUniformLocation("shadow");
		uniform_L = shader->GetUniformLocation("L");
	}
	
	free(shader_text_frag);
	return true;
}


//----------------- Shader text ----------------------------

char DRShaderFrameBuffer::DRSH_Vertex[] = "\n\
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

// Internal frame buffer compositing

// The internal frame buffer has a 12bit resolution per channel. 
// In order to accomodate an intensity range up to 4.0, all 
// color channels are divided by 4 before being saved to the
// frame buffer texture (actual 10bits per channel).
// When reading from the internal frame buffer, remember to:
// a) decompress the channels by multiplying the color by vec4(4,4,4,1)
// b) apply the same tone mapping as in the final post-processing stage.


char DRShaderFrameBuffer::DRSH_Framebuffer_Fragment_Header[] = "\n\
varying vec3 Necs; \n\
varying vec4 Pecs; \n\
uniform int width; \n\
uniform int height; \n\
uniform int use_gi; \n\
uniform vec3 ambient_term; \n\
uniform sampler2D RT_albedo; \n\
uniform sampler2D RT_normals; \n\
uniform sampler2D RT_lighting; \n\
uniform sampler2D RT_specular; \n\
uniform sampler2D RT_ao; \n\
uniform sampler2D RT_depth; \n\
uniform sampler2D noise; \n\
uniform sampler2D shadow; \n\
uniform vec3 sun_wcs; \n\
uniform vec3 camera_wcs; \n\
uniform float unitspm; \n\
uniform mat4  MVP_inverse; \n\
uniform mat4  Projection; \n\
uniform mat4  L;\n";

char DRShaderFrameBuffer::DRSH_Framebuffer_Fragment_Core[] = "\n\
void main(void) \n\
{ \n\
vec4 albedo = texture2D(RT_albedo,gl_TexCoord[0].st);\n\
vec4 lighting = vec4(2,2,2,1)*texture2D(RT_lighting,gl_TexCoord[0].st);\n\
vec4 normalbuf = texture2D(RT_normals,gl_TexCoord[0].st);\n\
vec3 normal = normalbuf.xyz;\n\
vec4 color; \n\
normal.x = normal.x*2-1;\n\
normal.y = normal.y*2-1;\n\
float emission = normalbuf.w;\n\
if (texture2D(RT_depth,gl_TexCoord[0].st).x<1.0) \n\
    emission*=2.0; \n\
vec4 ao = texture2D(RT_ao,gl_TexCoord[0].st);\n\
if (use_gi==1) \n\
lighting.xyz = lighting.xyz + ao.xyz + ambient_term; \n\
else \n\
	lighting.xyz = (lighting.xyz+ambient_term)*(1-ao.r); \n\
// We use a 12bit-per-channel framebuffer, so we can compress intensity \n\
// To be able ti store HDR color up to 4 times the clipping limit. \n\
color = vec4( (albedo*lighting).rgb + emission*albedo.rgb, 1.0); \n\
//color = vec4(ao.rgb,1.0); \n\
";

char DRShaderFrameBuffer::DRSH_Framebuffer_Fragment_Core_MultiSampledGI[] = "\n\
vec2 highPassDepth(in vec2 tc)\n\
{\n\
    float du=4.0/width; \n\
	float dv=4.0/height;\n\
	vec2 freq1 = vec2 (  abs(texture2D(RT_depth,tc+vec2(du,0.0)).r - texture2D(RT_depth,tc+vec2(-du,0.0)).r), \n\
	                     abs(texture2D(RT_depth,tc+vec2(0.0,dv)).r - texture2D(RT_depth,tc+vec2(0.0,-dv)).r) ); \n\
	return clamp( 50.0*freq1, vec2(0.0,0.0), vec2(1.0,1.0) );\n\
}\n\
void main(void) \n\
{ \n\
initGaussKernel(); \n\
vec4 albedo = texture2D(RT_albedo,gl_TexCoord[0].st);\n\
vec4 lighting = vec4(2,2,2,1)*texture2D(RT_lighting,gl_TexCoord[0].st);\n\
vec4 normalbuf = texture2D(RT_normals,gl_TexCoord[0].st);\n\
vec3 normal = normalbuf.xyz;\n\
vec4 color; \n\
normal.x = normal.x*2-1;\n\
normal.y = normal.y*2-1;\n\
float emission = normalbuf.w;\n\
if (texture2D(RT_depth,gl_TexCoord[0].st).x<1.0) \n\
    emission*=2.0; \n\
vec2 ksize = vec2(1.0,1.0)-highPassDepth(gl_TexCoord[0].st); \n\
ksize*=ksize; \n\
//vec2 offset = mix(5.0,0.7,ksize)*vec2(1.0/width,1.0/height); \n\
//vec2 offset = vec2 (mix(2.0,0.5,0.5*(sign(ksize.x-0.5)+1))/width, mix(5.0,0.7,0.5*(sign(ksize.y-0.5)+1))/height ); \n\
vec2 offset = vec2 (mix(1.0,5.0,ksize.x)/width, mix(0.5,5.0,ksize.y)/height ); \n\
vec4 ao = texture2D(RT_ao,gl_TexCoord[0].st);\n\
int i; \n\
vec4 samples[17]; \n\
samples[0] = ao; \n\
for (i=0; i<8; i++) \n\
{ \n\
	samples[i+1] = texture2D(RT_ao,gl_TexCoord[0].st \n\
	    +vec2(offset.x*kernel[2*i],offset.y*kernel[2*i+1])); \n\
	ao+=samples[i+1]; \n\
} \n\
for (i=8; i<16; i++) \n\
{ \n\
	samples[i+1] = texture2D(RT_ao,gl_TexCoord[0].st \n\
	    +vec2(offset.x*kernel[2*i],offset.y*kernel[2*i+1])); \n\
	ao+=0.5*samples[i+1]; \n\
} \n\
ao/=13; \n\
if (use_gi==1) \n\
	lighting.xyz = lighting.xyz + ao.xyz + ambient_term; \n\
else \n\
	lighting.xyz = (lighting.xyz+ambient_term)*(1-ao.r); \n\
// We use a 12bit-per-channel framebuffer, so we can compress intensity \n\
// To be able ti store HDR color up to 4 times the clipping limit. \n\
color = (vec4((albedo*lighting).rgb + emission*albedo.rgb,1.0)); \n\
//color = vec4(0.2*length(offset)*vec3(1,1,1),1.0); \n\
";

char DRShaderFrameBuffer::DRSH_Framebuffer_Fragment_Footer[] = "\n\
gl_FragColor = clamp(vec4(0.25,0.25,0.25,1.0)*color,vec4(0.0,0.0,0.0,0.0),vec4(1.0,1.0,1.0,1.0)); \n\
}";
