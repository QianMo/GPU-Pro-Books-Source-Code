
#include "shaders/DeferredRendererShader_GI_IV_Injection_SS.h"
#include "cglibdefines.h"

void DRShaderGI_IV_Injection_SS::start()
{
	DRShaderGI::start();

	shader->begin();

	shader->setUniform1i(0,0, uniform_inj_zbuffer);
	shader->setUniform1i(0,1, uniform_inj_albedo);
	shader->setUniform1i(0,2, uniform_inj_normals);
	shader->setUniform1i(0,3, uniform_inj_lighting);

	shader->setUniform1i(0, vol_depth, uniform_inj_depth);
	shader->setUniform1i(0, prebaked_lighting, uniform_inj_prebaked_lighting);

//	shader->setUniformMatrix4fv(0,1, false, fmat_MV, uniform_inj_MV);
	shader->setUniformMatrix4fv(0,1, false, fmat_P, uniform_inj_P);
	shader->setUniformMatrix4fv(0,1, false, fmat_MVP_inv, uniform_inj_MVP_inverse);
}

bool DRShaderGI_IV_Injection_SS::init(class DeferredRenderer* _renderer)
{
	if (initialized)
		return true;

	if (!DRShaderGI::init(_renderer))
		return false;

	char * shader_text_vert = DRSH_GI_IV_Vert;
	char * shader_text_geom = DRSH_GI_IV_Geom;
	char * shader_text_frag = DRSH_GI_IV_Frag;

	shader = shader_manager.loadfromMemory ("Global Illumination Injection SS Vert", shader_text_vert, 
		                                    "Global Illumination Injection SS Geom", shader_text_geom, 
											"Global Illumination Injection SS Frag", shader_text_frag);
	if (!shader)
	{
		EAZD_TRACE ("DeferredRenderer::buildShaders() : ERROR - Problem compiling GI shader (IV Inject SS).\n");
		return false;
	}
	else
	{
		uniform_inj_zbuffer		= shader->GetUniformLocation("zbuffer");
		uniform_inj_albedo		= shader->GetUniformLocation("albedo");
		uniform_inj_normals		= shader->GetUniformLocation("normals");
		uniform_inj_lighting	= shader->GetUniformLocation("lighting");

		uniform_inj_depth		= shader->GetUniformLocation("vol_depth");
		uniform_inj_prebaked_lighting = shader->GetUniformLocation("prebaked_lighting");

	//	uniform_inj_MV			= shader->GetUniformLocation("MV");
		uniform_inj_P			= shader->GetUniformLocation("P");
		uniform_inj_MVP_inverse	= shader->GetUniformLocation("MVP_inverse");
	}
	
	initialized = true;
	return true;
}

//----------------- Shader text ----------------------------

char DRShaderGI_IV_Injection_SS::DRSH_GI_IV_Vert[] = "\n\
 \n\
#version 330 compatibility \n\
 \n\
// --- NVidia GLSL compiling options \n\
#pragma optionNV(fastmath off) \n\
#pragma optionNV(fastprecision off) \n\
#pragma optionNV(ifcvt none) \n\
#pragma optionNV(inline all) \n\
#pragma optionNV(strict on) \n\
#pragma optionNV(unroll all) \n\
 \n\
flat out vec2 tex_coord; \n\
 \n\
uniform sampler2D zbuffer; \n\
 \n\
// MVP_inverse Matrix is the camera's matrix \n\
uniform mat4 MVP_inverse; \n\
 \n\
// Convert Canonical Screen Space to Object Space \n\
vec3 PointCSS2WCS (in vec3 sample) \n\
{ \n\
	vec4 point_wcs = MVP_inverse*vec4(sample,1); \n\
	return point_wcs.xyz / point_wcs.w; \n\
} \n\
 \n\
vec4 SampleBuffer(sampler2D buffer, vec2 uv) \n\
{ \n\
    ivec2 texel_coord = ivec2(floor(uv*textureSize(buffer,0)-vec2(0.48,0.48))); \n\
    return texelFetch (buffer,texel_coord,0); \n\
} \n\
 \n\
void main (void) \n\
{ \n\
	tex_coord = gl_Vertex.xy; \n\
	float depth = SampleBuffer (zbuffer, tex_coord).x; \n\
 \n\
  	vec3 pos_css = 2*vec3 (gl_Vertex.xy, depth)-vec3(1,1,1); \n\
	vec3 pos_wcs = PointCSS2WCS (pos_css); \n\
 \n\
	gl_Position = gl_ModelViewProjectionMatrix * vec4 (pos_wcs, 1.0);	// world space --> clip space \n\
}";

char DRShaderGI_IV_Injection_SS::DRSH_GI_IV_Frag[] = "\n\
 \n\
#version 330 compatibility \n\
 \n\
// --- NVidia GLSL compiling options \n\
#pragma optionNV(fastmath off) \n\
#pragma optionNV(fastprecision off) \n\
#pragma optionNV(ifcvt none) \n\
#pragma optionNV(inline all) \n\
#pragma optionNV(strict on) \n\
#pragma optionNV(unroll all) \n\
 \n\
#define MAP_0TO1_MINUS1TO1(_value)	(2.0 * (_value) - 1.0)	// Map [0.0,1.0] to [-1.0,1.0] \n\
 \n\
// MVP_inverse & P Matrix is the camera's matrix \n\
uniform mat4 MVP_inverse, MV, P; \n\
 \n\
vec3 VectorECS2WCS(in vec3 sample) \n\
{ \n\
	vec4 vector_WCS = MVP_inverse*P*vec4(sample,1); \n\
	vec4 zero_WCS = MVP_inverse*P*vec4(0,0,0,1); \n\
	vector_WCS = vector_WCS/vector_WCS.w-zero_WCS/zero_WCS.w; \n\
	return vector_WCS.xyz; \n\
} \n\
 \n\
flat in vec2 gtex_coord; \n\
 \n\
uniform sampler2D zbuffer, albedo, normals, lighting; \n\
uniform float one_over_vol_width, one_over_vol_height; \n\
uniform int prebaked_lighting; \n\
 \n\
vec4 sh_basis (const in vec3 dir) \n\
{ \n\
    float   L00  = 0.282094792; \n\
    float   L1_1 = 0.488602512 * dir.y; \n\
    float   L10  = 0.488602512 * dir.z; \n\
    float   L11  = 0.488602512 * dir.x; \n\
 \n\
    // sh is in [-1,1] range \n\
    return vec4 (L11, L1_1, L10, L00); \n\
} \n\
 \n\
vec4 SampleBuffer(sampler2D buffer, vec2 uv) \n\
{ \n\
    ivec2 texel_coord = ivec2(floor(uv*textureSize(buffer,0)-vec2(0.48,0.48))); \n\
    return texelFetch (buffer, texel_coord,0); \n\
} \n\
 \n\
void main (void) \n\
{ \n\
	vec4 _albedo = SampleBuffer(albedo, gtex_coord);			// read the albedo buffer \n\
 \n\
	vec3 _normals_ecs = SampleBuffer(normals, gtex_coord).xyz;	// read the normals buffer \n\
	_normals_ecs.x = MAP_0TO1_MINUS1TO1 (_normals_ecs.x); \n\
	_normals_ecs.y = MAP_0TO1_MINUS1TO1 (_normals_ecs.y); \n\
	_normals_ecs = normalize (_normals_ecs); \n\
	vec3 _normals_wcs = VectorECS2WCS (_normals_ecs); \n\
	_normals_wcs = normalize (_normals_wcs); \n\
 \n\
	vec4 _lighting = vec4 (2,2,2,1) * SampleBuffer (lighting, gtex_coord);		// read the lighting buffer \n\
 \n\
	vec4 _color = (prebaked_lighting == 1) ? _albedo : _albedo * _lighting; \n\
	vec4 shb = sh_basis (_normals_wcs); \n\
 \n\
	gl_FragData[0] = vec4 (_normals_wcs, 1.0); \n\
	gl_FragData[1] = _color.r * shb; \n\
	gl_FragData[2] = _color.g * shb; \n\
	gl_FragData[3] = _color.b * shb; \n\
}";

char DRShaderGI_IV_Injection_SS::DRSH_GI_IV_Geom[] = "\n\
 \n\
#version 330 compatibility \n\
#extension GL_EXT_geometry_shader4 : enable \n\
 \n\
// --- NVidia GLSL compiling options \n\
#pragma optionNV(fastmath off) \n\
#pragma optionNV(fastprecision off) \n\
#pragma optionNV(ifcvt none) \n\
#pragma optionNV(inline all) \n\
#pragma optionNV(strict on) \n\
#pragma optionNV(unroll all) \n\
 \n\
layout(points) in; \n\
layout(points, max_vertices = 1) out; \n\
 \n\
uniform int vol_depth; \n\
 \n\
flat in vec2 tex_coord[]; \n\
flat out vec2 gtex_coord; \n\
 \n\
#define MAP_MINUS1TO1_0TO1(_value)	(0.5 * (_value) + 0.5)	// Map [-1.0,1.0] to [0.0,1.0] \n\
 \n\
void main (void) \n\
{ \n\
	gtex_coord = tex_coord[0]; \n\
 \n\
	gl_Position = gl_PositionIn[0]; \n\
	gl_Layer = int (vol_depth * MAP_MINUS1TO1_0TO1 (gl_Position.z)); \n\
 \n\
	EmitVertex(); \n\
}";
