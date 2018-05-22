
#include "shaders/DeferredRendererShader_GI_IV_Injection_SS_Cleanup.h"
#include "cglibdefines.h"

void DRShaderGI_IV_Injection_SS_Cleanup::start()
{
	DRShaderGI::start();

	shader->begin();

	shader->setUniform1i(0,0, uniform_cln_zbuffer);
	shader->setUniform1i(0,1, uniform_cln_albedo);
	shader->setUniform1i(0,2, uniform_cln_normals);
	shader->setUniform1i(0,3, uniform_cln_lighting);

	shader->setUniform1i(0,4, uniform_cln_vol_normals);
	shader->setUniform1i(0,5, uniform_cln_vol_shR);
	shader->setUniform1i(0,6, uniform_cln_vol_shG);
	shader->setUniform1i(0,7, uniform_cln_vol_shB);
	shader->setUniform1i(0,8, uniform_cln_vol_zbuffer);
//	shader->setUniform1i(0,X, uniform_cln_vol_albedo);
//	shader->setUniform1i(0,X, uniform_cln_vol_lighting);
//	shader->setUniform1i(0,X, uniform_cln_vol_color);

	shader->setUniform1f(0, 1.0f / (float) vol_width,  uniform_cln_vol_width);
	shader->setUniform1f(0, 1.0f / (float) vol_height, uniform_cln_vol_height);

	shader->setUniform1f(0, voxel_radius, uniform_cln_voxel_radius);
	shader->setUniform3f(0, voxel_half_size[0], voxel_half_size[1], voxel_half_size[2], uniform_cln_voxel_half_size);

	shader->setUniform1i(0, prebaked_lighting, uniform_cln_prebaked_lighting);

	shader->setUniformMatrix4fv(0,1, false, fmat_P, uniform_cln_P);
	shader->setUniformMatrix4fv(0,1, false, fmat_P_inv, uniform_cln_P_inverse);
	shader->setUniformMatrix4fv(0,1, false, fmat_MVP, uniform_cln_MVP);
	shader->setUniformMatrix4fv(0,1, false, fmat_MVP_inv, uniform_cln_MVP_inverse);

	shader->setUniform3f(0, cop[0], cop[1], cop[2], uniform_cop);
}

bool DRShaderGI_IV_Injection_SS_Cleanup::init(class DeferredRenderer* _renderer)
{
	if (initialized)
		return true;

	if (!DRShaderGI::init(_renderer))
		return false;

	char * shader_text_vert = DRSH_GI_IV_Vert;
	char * shader_text_frag = DRSH_GI_IV_Frag;

	shader = shader_manager.loadfromMemory ("Global Illumination Injection SS Cleanup Vert", shader_text_vert,
											"Global Illumination Injection SS Cleanup Frag", shader_text_frag);
	if (!shader)
	{
		EAZD_TRACE ("DeferredRenderer::buildShaders() : ERROR - Problem compiling GI shader (IV Inject SS Cleanup).\n");
		return false;
	}
	else
	{
		uniform_cln_vol_width	= shader->GetUniformLocation("one_over_vol_width");
		uniform_cln_vol_height	= shader->GetUniformLocation("one_over_vol_height");

		uniform_cln_zbuffer		= shader->GetUniformLocation("zbuffer");
		uniform_cln_albedo		= shader->GetUniformLocation("albedo");
		uniform_cln_normals		= shader->GetUniformLocation("normals");
		uniform_cln_lighting	= shader->GetUniformLocation("lighting");

		uniform_cln_vol_shR			= shader->GetUniformLocation("vol_shR");
		uniform_cln_vol_shG			= shader->GetUniformLocation("vol_shG");
		uniform_cln_vol_shB			= shader->GetUniformLocation("vol_shB");
		uniform_cln_vol_normals		= shader->GetUniformLocation("vol_normals");
		uniform_cln_vol_zbuffer		= shader->GetUniformLocation("vol_zbuffer");
	//	uniform_cln_vol_albedo		= shader->GetUniformLocation("vol_albedo");
	//	uniform_cln_vol_lighting	= shader->GetUniformLocation("vol_lighting");
	//	uniform_cln_vol_color		= shader->GetUniformLocation("vol_color");

		uniform_cln_voxel_radius	= shader->GetUniformLocation("voxel_radius");
		uniform_cln_voxel_half_size	= shader->GetUniformLocation("voxel_half_size");
		uniform_cln_prebaked_lighting = shader->GetUniformLocation("prebaked_lighting");

		uniform_cln_P			= shader->GetUniformLocation("P");
		uniform_cln_P_inverse	= shader->GetUniformLocation("P_inverse");
		uniform_cln_MVP			= shader->GetUniformLocation("MVP");
		uniform_cln_MVP_inverse	= shader->GetUniformLocation("MVP_inverse");

		uniform_cop	        = shader->GetUniformLocation("cop");
	}
	
	initialized = true;
	return true;
}

//----------------- Shader text ----------------------------

char DRShaderGI_IV_Injection_SS_Cleanup::DRSH_GI_IV_Vert[] = "\n\
 \n\
#version 330 compatibility \n\
 \n\
// --- NVidia GLSL compiling options \n\
#pragma optionNV(fastmath on) \n\
#pragma optionNV(fastprecision on) \n\
#pragma optionNV(ifcvt none) \n\
#pragma optionNV(inline all) \n\
#pragma optionNV(strict on) \n\
#pragma optionNV(unroll all) \n\
 \n\
out vec3 voxel_tex_coord, voxel_position; \n\
 \n\
void main (void) \n\
{ \n\
	voxel_position = gl_MultiTexCoord0.xyz; \n\
	voxel_tex_coord = vec3(0.5*gl_Vertex.xy+vec2(0.5), 0.5+0.5*gl_Vertex.z); \n\
 \n\
	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex; \n\
}";

char DRShaderGI_IV_Injection_SS_Cleanup::DRSH_GI_IV_Frag[] = "\n\
 \n\
#version 330 compatibility \n\
 \n\
// --- NVidia GLSL compiling options \n\
#pragma optionNV(inline all) \n\
#pragma optionNV(strict on) \n\
#pragma optionNV(unroll all) \n\
 \n\
#define MAP_MINUS1TO1_0TO1(_value)	(0.5 * (_value) + 0.5)	// Map [-1.0,1.0] to [0.0,1.0] \n\
#define MAP_0TO1_MINUS1TO1(_value)	(2.0 * (_value) - 1.0)	// Map [0.0,1.0] to [-1.0,1.0] \n\
 \n\
in vec3 voxel_position; \n\
in vec3 voxel_tex_coord; \n\
 \n\
uniform float one_over_vol_width, one_over_vol_height; \n\
uniform float voxel_radius; \n\
uniform vec3 cop, voxel_half_size; \n\
 \n\
uniform sampler2D zbuffer, albedo, normals, lighting; \n\
uniform sampler3D vol_shR, vol_shG, vol_shB, vol_normals; \n\
 \n\
uniform mat4 MVP, MVP_inverse, P, P_inverse; \n\
 \n\
uniform int prebaked_lighting; \n\
 \n\
vec3 PointWCS2CSS (in vec3 sample) \n\
{ \n\
    vec4 point_css = MVP*vec4(sample,1); \n\
    return point_css.xyz / point_css.w; \n\
} \n\
 \n\
vec3 PointWCS2ECS (in vec3 sample) \n\
{ \n\
    vec4 point_ecs = P_inverse*MVP*vec4(sample,1); \n\
    return point_ecs.xyz / point_ecs.w; \n\
} \n\
 \n\
vec3 PointCSS2WCS (in vec3 sample) \n\
{ \n\
    vec4 point_wcs = MVP_inverse*vec4(sample,1); \n\
    return point_wcs.xyz / point_wcs.w; \n\
} \n\
 \n\
vec3 PointCSS2ECS (in vec3 sample) \n\
{ \n\
    vec4 point_ecs = P_inverse*vec4(sample,1); \n\
    return point_ecs.xyz / point_ecs.w; \n\
} \n\
 \n\
vec3 VectorECS2WCS(in vec3 sample) \n\
{ \n\
    vec4 vector_WCS = MVP_inverse*P*vec4(sample,1); \n\
    vec4 zero_WCS = MVP_inverse*P*vec4(0,0,0,1); \n\
    vector_WCS = vector_WCS/vector_WCS.w-zero_WCS/zero_WCS.w; \n\
    return vector_WCS.xyz; \n\
} \n\
 \n\
vec4 SampleBuffer(sampler2D buffer, vec2 uv) \n\
{ \n\
    ivec2 texel_coord = ivec2(floor(uv*textureSize(buffer,0)-vec2(0.0,0.0))); \n\
    return texelFetch (buffer,texel_coord,0); \n\
} \n\
 \n\
vec4 SampleBuffer(sampler3D buffer, vec3 uvw) \n\
{ \n\
    ivec3 texel_coord = ivec3(floor(uvw*textureSize(buffer,0)-vec3(0.48,0.48,0.48))); \n\
    return texelFetch (buffer,texel_coord,0); \n\
} \n\
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
void main (void) \n\
{ \n\
	vec4 voxel_pos_wcs = vec4(voxel_position, 1.0); \n\
	vec3 voxel_pos_css = PointWCS2CSS (voxel_pos_wcs.xyz); \n\
	vec3 voxel_pos_ecs = PointWCS2ECS (voxel_pos_wcs.xyz); \n\
 	vec3 voxel_pos_ss = vec3(0.5 * voxel_pos_css.xyz + vec3(0.5));	// Map [-1.0,1.0] to [0.0,1.0] \n\
\n\
	float depth	= SampleBuffer (zbuffer, voxel_pos_ss.xy).x; \n\
	vec3 zbuffer_css = vec3 (voxel_pos_css.xy, 2.0*depth-1.0); \n\
	vec3 zbuffer_ecs = PointCSS2ECS (zbuffer_css); \n\
 \n\
	vec3 voxel_middlefront_wcs	= voxel_pos_wcs.xyz + voxel_radius * vec3 (1.0); \n\
	voxel_middlefront_wcs = max (voxel_middlefront_wcs, voxel_pos_wcs.xyz + voxel_half_size); \n\
 \n\
	vec3 voxel_middleback_wcs	= voxel_pos_wcs.xyz - voxel_radius * vec3 (1.0); \n\
	voxel_middleback_wcs = min (voxel_middleback_wcs, voxel_pos_wcs.xyz - voxel_half_size); \n\
 \n\
	vec3 voxel_middlefront_css	= PointWCS2CSS (voxel_middlefront_wcs); \n\
	vec3 voxel_middleback_css	= PointWCS2CSS (voxel_middleback_wcs); \n\
	vec3 voxel_middlefront_ecs	= PointWCS2ECS (voxel_middlefront_wcs); \n\
	vec3 voxel_middleback_ecs	= PointWCS2ECS (voxel_middleback_wcs); \n\
 \n\
	float bias = 0.5 * distance (voxel_middlefront_ecs, voxel_middleback_ecs); \n\
 \n\
	vec4 shR_value, shG_value, shB_value, normal_value, zbuffer_value; \n\
	shR_value = SampleBuffer (vol_shR, voxel_tex_coord); \n\
	shG_value = SampleBuffer (vol_shG, voxel_tex_coord); \n\
	shB_value = SampleBuffer (vol_shB, voxel_tex_coord); \n\
	normal_value = SampleBuffer (vol_normals, voxel_tex_coord); \n\
 \n\
	// keep voxels outside of frustum \n\
	if ((any (lessThan (voxel_pos_css, vec3(-1)))) || (any (greaterThan (voxel_pos_css, vec3(1))))) \n\
	{ \n\
		zbuffer_value	= vec4(1,1,1,1); \n\
		 \n\
		gl_FragData[0] = normal_value; \n\
		gl_FragData[1] = shR_value; \n\
		gl_FragData[2] = shG_value; \n\
		gl_FragData[3] = shB_value; \n\
		return; \n\
	} \n\
 \n\
	// discard \n\
	if (voxel_pos_ecs.z > zbuffer_ecs.z + bias) \n\
	{ \n\
		normal_value	= vec4 (0,0,0,0); \n\
		shR_value		= \n\
		shG_value		= \n\
		shB_value		= vec4 (0,0,0,0); \n\
	} \n\
 \n\
	gl_FragData[0] = normal_value; \n\
	gl_FragData[1] = shR_value; \n\
	gl_FragData[2] = shG_value; \n\
	gl_FragData[3] = shB_value; \n\
}";
