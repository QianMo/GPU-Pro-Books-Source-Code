
#include "shaders/DeferredRendererShader_GI_IV.h"
#include "DeferredRenderer.h"

void DRShaderGI_IV::start()
{
	int view[4];
	glGetIntegerv(GL_VIEWPORT,view);

	Vector3D voxelsize = bbox.getSize();
    voxelsize.x /= dim[0];
    voxelsize.y /= dim[1];
    voxelsize.z /= dim[2];

	DRShaderGI::start();

	shader->begin();

//	shader->setUniform1i(0,view[2],uniform_width);
//	shader->setUniform1i(0,view[3],uniform_height);
//	shader->setUniform1f(0,renderer->getGIRenderer()->getRange(),uniform_R_wcs);
	shader->setUniform1i(0,4,uniform_RT_normals);
	shader->setUniform1i(0,5,uniform_RT_depth);
//	shader->setUniform1i(0,6,uniform_Noise);
	shader->setUniform1i(0,7,uniform_photonmap_red);
	shader->setUniform1i(0,8,uniform_photonmap_green);
	shader->setUniform1i(0,9,uniform_photonmap_blue);
	shader->setUniformMatrix4fv(0,1,false,fmat_MVP_inv,uniform_MVP_inverse);
	shader->setUniformMatrix4fv(0,1,false,fmat_MVP,uniform_MVP);
	shader->setUniformMatrix4fv(0,1,false,fmat_P,uniform_Projection);
//	shader->setUniformMatrix4fv(0,1,false,fmat_P_inv,uniform_Projection_inverse);
	shader->setUniform1f(0, renderer->getGIRenderer()->getFactor(), uniform_factor);	   
	shader->setUniform3f(0, (float) dim[0], (float) dim[1], (float) dim[2], uniform_photonmap_res);
//	shader->setUniform1f(0,voxelsize.length(),uniform_voxelsize);
	shader->setUniform1i(0,renderer->getGIRenderer()->getNumSamples(),uniform_samples);
}

bool DRShaderGI_IV::init(class DeferredRenderer* _renderer)
{
	if (initialized)
		return true;

	if (!DRShaderGI::init(_renderer))
		return false;
	
	char * shader_text_vert;
    char * shader_text_frag;

	shader_text_vert = DRSH_GI_IV_Vert;
	shader_text_frag = (char*)malloc(20000);
	memset(shader_text_frag, 0, 20000);
	strcpy(shader_text_frag,DRSH_GI_IV_Frag_Header);
	strcat(shader_text_frag,DRSH_GI_IV_Frag_SH);
	strcat(shader_text_frag,DRSH_GI_IV_Frag_Main);

	shader = shader_manager.loadfromMemory ("Global Illumination Draw Vert", shader_text_vert,
		                                    "Global Illumination Draw Frag", shader_text_frag);
	if (!shader)
	{
		EAZD_TRACE ("DeferredRenderer::buildShaders() : ERROR - Problem compiling GI shader.\n");
		return false;
	}
	else
	{
    //  uniform_width = shader->GetUniformLocation("width");
    //  uniform_height = shader->GetUniformLocation("height");
        uniform_RT_normals = shader->GetUniformLocation("RT_normals");
        uniform_RT_depth = shader->GetUniformLocation("RT_depth");
    //  uniform_Noise = shader->GetUniformLocation("Noise");
        uniform_photonmap_red = shader->GetUniformLocation("photonmap_red");
		uniform_photonmap_green = shader->GetUniformLocation("photonmap_green");
		uniform_photonmap_blue = shader->GetUniformLocation("photonmap_blue");
        uniform_MVP_inverse = shader->GetUniformLocation("MVP_inverse");
        uniform_MVP = shader->GetUniformLocation("MVP");
    //  uniform_Projection_inverse = shader->GetUniformLocation("Projection_inverse");
        uniform_Projection = shader->GetUniformLocation("Projection");
    //  uniform_R_wcs = shader->GetUniformLocation("R_wcs");
    //  uniform_voxelsize = shader->GetUniformLocation("voxelsize");
        uniform_samples = shader->GetUniformLocation("samples");
        uniform_factor = shader->GetUniformLocation("factor");
		uniform_photonmap_res = shader->GetUniformLocation("photonmap_res");
	}
	
	free(shader_text_frag);
	initialized = true;
	return true;
}

//----------------- Shader text ----------------------------

char DRShaderGI_IV::DRSH_GI_IV_Vert[] = "\n\
void main(void) \n\
{ \n\
   gl_Position = ftransform(); \n\
   gl_TexCoord[0] = gl_TextureMatrix[0]*gl_MultiTexCoord0; \n\
   gl_TexCoord[1] = gl_TextureMatrix[1]*gl_MultiTexCoord1; \n\
   gl_TexCoord[2] = gl_TextureMatrix[2]*gl_MultiTexCoord2; \n\
} \n\
";

char DRShaderGI_IV::DRSH_GI_IV_Frag_Header[] = "\n\
uniform int   width; \n\
uniform int   height; \n\
uniform sampler2D RT_normals; \n\
uniform sampler2D RT_depth; \n\
//uniform sampler3D Noise; \n\
uniform sampler3D photonmap_red; \n\
uniform sampler3D photonmap_green; \n\
uniform sampler3D photonmap_blue; \n\
uniform mat4  MVP_inverse; \n\
uniform mat4  MVP; \n\
uniform mat4  Projection_inverse; \n\
uniform mat4  Projection; \n\
uniform float R_wcs; \n\
uniform float voxelsize; \n\
uniform int samples; \n\
uniform float factor; \n\
uniform vec3 photonmap_res; \n\
\n\
#define THREE_GI_BANDS \n\
//#define POINT_FILTER \n\
//#define TRILINEAR_FILTER \n\
//#define REJECTION_FILTER \n\
#define NORMAL_FILTER6 \n\
\n\
vec3 VectorWCS2CSS(in vec3 sample) \n\
{ \n\
vec4 vector_CSS = MVP*vec4(sample,1); \n\
vec4 zero_CSS = MVP*vec4(0,0,0,1); \n\
vector_CSS=vector_CSS/vector_CSS.w-zero_CSS/zero_CSS.w; \n\
return vector_CSS.xyz; \n\
} \n\
vec3 PointWCS2CSS(in vec3 sample) \n\
{ \n\
    vec4 p_css = MVP*vec4(sample,1); \n\
	return p_css.xyz/p_css.w; \n\
} \n\
\n\
vec3 VectorECS2WCS(in vec3 sample) \n\
{ \n\
vec4 vector_WCS = MVP_inverse*Projection*vec4(sample,1); \n\
vec4 zero_WCS = MVP_inverse*Projection*vec4(0,0,0,1); \n\
vector_WCS=vector_WCS/vector_WCS.w-zero_WCS/zero_WCS.w; \n\
return vector_WCS.xyz; \n\
} \n\
\n\
vec3 PointCSS2WCS(in vec3 sample) \n\
{ \n\
    vec4 p_wcs = MVP_inverse*vec4(sample,1); \n\
	return p_wcs.xyz/p_wcs.w; \n\
} \n\
\n\
vec4 mymix(in vec4 a, in vec4 b, float t) \n\
{ \n\
	if(a.w<0.0) \n\
		a=b; \n\
	else if (b.w<0.0) \n\
		b= a; \n\
	return mix(a,b,t); \n\
} \n\
\n\
vec4 bfilter(sampler3D tex, vec3 pos_ws, vec3 normal) \n\
{ \n\
	vec3 pos = vec3(gl_TextureMatrix[1]* vec4(pos_ws,1.0)); \n\
#ifdef TRILINEAR_FILTER \n\
	float cx = 1./photonmap_res.x; \n\
	float cy = 1./photonmap_res.y; \n\
	float cz = 1./photonmap_res.z; \n\
	\n\
	vec4 x1 = 2.0*texture3D(tex, pos + vec3(0,0,0))-1.0; \n\
	vec4 x2 = 2.0*texture3D(tex, pos + vec3(cx,0 ,0))-1.0; \n\
	vec4 x3 = 2.0*texture3D(tex, pos + vec3(0 ,cy,0))-1.0; \n\
	vec4 x4 = 2.0*texture3D(tex, pos + vec3(cx,cy,0))-1.0; \n\
	vec4 y1 = 2.0*texture3D(tex, pos + vec3(0 ,0,cz))-1.0; \n\
	vec4 y2 = 2.0*texture3D(tex, pos + vec3(cx,0,cz))-1.0; \n\
	vec4 y3 = 2.0*texture3D(tex, pos + vec3(0 ,cy,cz))-1.0; \n\
	vec4 y4 = 2.0*texture3D(tex, pos + vec3(cx,cy,cz))-1.0; \n\
\n\
	float dx = fract(pos.x* photonmap_res.x); \n\
	float dy = fract(pos.y* photonmap_res.y); \n\
	float dz = fract(pos.z* photonmap_res.z); \n\
\n\
	vec4 m1 = mix(x1, x2, dx); \n\
	vec4 m2 = mix(x3, x4, dx); \n\
	vec4 m3 = mix(m1, m2, dy); \n\
	vec4 m4 = mix(y1, y2, dx); \n\
	vec4 m5 = mix(y3, y4, dx); \n\
	vec4 m6 = mix(m4, m5, dy); \n\
	vec4 res = mix(m3, m6, dz); \n\
\n\
	return res; \n\
#endif \n\
\n\
#ifdef REJECTION_FILTER \n\
	float cx = 1./photonmap_res.x; \n\
	float cy = 1./photonmap_res.y; \n\
	float cz = 1./photonmap_res.z; \n\
\n\
	vec4 x1 = 2.0*texture3D(tex, pos + vec3(0,0,0))-1.0; \n\
	vec4 x2 = 2.0*texture3D(tex, pos + vec3(cx,0 ,0))-1.0; \n\
	vec4 x3 = 2.0*texture3D(tex, pos + vec3(0 ,cy,0))-1.0; \n\
	vec4 x4 = 2.0*texture3D(tex, pos + vec3(cx,cy,0))-1.0; \n\
	vec4 y1 = 2.0*texture3D(tex, pos + vec3(0 ,0,cz))-1.0; \n\
	vec4 y2 = 2.0*texture3D(tex, pos + vec3(cx,0,cz))-1.0; \n\
	vec4 y3 = 2.0*texture3D(tex, pos + vec3(0 ,cy,cz))-1.0; \n\
	vec4 y4 = 2.0*texture3D(tex, pos + vec3(cx,cy,cz))-1.0; \n\
\n\
	float dx = fract(pos.x* photonmap_res.x); \n\
	float dy = fract(pos.y* photonmap_res.y); \n\
	float dz = fract(pos.z* photonmap_res.z); \n\
\n\
	vec4 m1 = mymix(x1, x2, dx); \n\
	vec4 m2 = mymix(x3, x4, dx); \n\
	vec4 m3 = mymix(m1, m2, dy); \n\
	vec4 m4 = mymix(y1, y2, dx); \n\
	vec4 m5 = mymix(y3, y4, dx); \n\
	vec4 m6 = mymix(m4, m5, dy); \n\
	vec4 res = mymix(m3, m6, dz); \n\
\n\
	return res; \n\
#endif \n\
\n\
#ifdef NORMAL_FILTER6 \n\
\n\
float cx = 1./(photonmap_res.x); \n\
float cy = 1./(photonmap_res.y); \n\
float cz = 1./(photonmap_res.z); \n\
\n\
	pos +=  1.2*(1./photonmap_res ) * normal; \n\
\n\
	vec4 p1 = texture3D(tex, pos + vec3(0,cy,0)); \n\
	vec4 p2 = texture3D(tex, pos + vec3(0,-cy,0)); \n\
	vec4 p3 = texture3D(tex, pos + vec3(cx,0,0)); \n\
	vec4 p4 = texture3D(tex, pos + vec3(-cx,0,0)); \n\
	vec4 p5 = texture3D(tex, pos + vec3(0,0,cz)); \n\
	vec4 p6 = texture3D(tex, pos + vec3(0,0,-cz)); \n\
	vec4 p7 = texture3D(tex, pos + vec3(0,0,0)); \n\
	/* \n\
	float w1 = max(0.0,dot(normal, vec3(0,1,0))); \n\
	float w2 = max(0.0,dot(normal, vec3(0,-1,0))); \n\
	float w3 = max(0.0,dot(normal, vec3(1,0,0))); \n\
	float w4 = max(0.0,dot(normal, vec3(-1,0,0))); \n\
	float w5 = max(0.0,dot(normal, vec3(0,0,1))); \n\
	float w6 = max(0.0,dot(normal, vec3(0,0,-1))); \n\
	 */ \n\
	float w1 = max(0.4,normal.y); \n\
	float w2 = max(0.4,-normal.y); \n\
	float w3 = max(0.4,normal.x); \n\
	float w4 = max(0.4,-normal.x); \n\
	float w5 = max(0.4,normal.z); \n\
	float w6 = max(0.4,-normal.z); \n\
	\n\
	vec4 res = (w1*p1+w2*p2+w3*p3+w4*p4+w5*p5+w6*p6)/(w1+w2+w3+w4+w5+w6); \n\
	\n\
	return res; \n\
#endif \n\
\n\
#ifdef POINT_FILTER \n\
	return texture3D(tex, pos); \n\
#endif \n\
}"; 

char DRShaderGI_IV::DRSH_GI_IV_Frag_SH[] = "\n\
vec4 sh_basis (const in vec3 dir) \n\
{ \n\
    float   L00  = 0.282094792; \n\
    float   L1_1 = 0.488602512 * dir.y; \n\
    float   L10  = 0.488602512 * dir.z; \n\
    float   L11  = 0.488602512 * dir.x; \n\
 \n\
    // sh is in [-1,1] range \n\
    return vec4 (L11, L1_1, L10, L00); \n\
}\n\
vec4 sh_encode(vec3 dir, float val) \n\
{ \n\
	return val*sh_basis(dir); \n\
} \n\
\n\
float sh_decode(vec4 sh, vec3 dir) \n\
{ \n\
	return dot(sh, sh_basis(dir)); \n\
} \n\
\n\
vec4 sh_coslobe (vec3 dir) \n\
{ \n\
	dir = normalize(dir); \n\
	return vec4( 1.023326*dir.x, 1.023326*dir.y, 1.023326*dir.z, 0.886226); \n\
}; \n\
// returns the illumination integral over the hemisphere defined by dir \n\
// int_{Omega}{Lcos theta d omega} \n\
float sh_cos_integral(vec4 sh, vec3 dir) \n\
{ \n\
	return dot(sh_coslobe(dir),sh); \n\
  \n\
}\n\
\n";

char DRShaderGI_IV::DRSH_GI_IV_Frag_Main[] = "\n\
void main(void) \n\
{ \n\
	int num_samples = 10; \n\
	float depth = texture2D(RT_depth,gl_TexCoord[0].st).r; \n\
	if (depth==1.0) \n\
	{ \n\
		gl_FragColor = vec4(0,0,0,1);//discard; \n\
		return ;\n\
	} \n\
	vec3 pos_css = vec3(2.0*gl_TexCoord[0].x-1.0, 2.0*gl_TexCoord[0].y-1.0, 2*depth-1.0); \n\
	vec3 normal_ecs = texture2D(RT_normals,gl_TexCoord[0].st).xyz; \n\
	normal_ecs.x = normal_ecs.x*2.0-1.0; \n\
	normal_ecs.y = normal_ecs.y*2.0-1.0; \n\
	\n\
	normal_ecs = normalize(normal_ecs); \n\
	vec3 normal_wcs = VectorECS2WCS(normal_ecs); \n\
	\n\
	normal_wcs = normalize(normal_wcs); \n\
	vec3 pos_wcs = PointCSS2WCS(pos_css); \n\
	\n\
	vec4 shr = bfilter(photonmap_red,pos_wcs, normal_wcs); \n\
	float Lr = sh_cos_integral(shr, -normal_wcs ); \n\
	\n\
	vec4 shg = bfilter(photonmap_green,pos_wcs, normal_wcs); \n\
	float Lg = sh_cos_integral(shg, -normal_wcs ); \n\
	\n\
	vec4 shb = bfilter(photonmap_blue,pos_wcs, normal_wcs); \n\
	float Lb = sh_cos_integral(shb, -normal_wcs ); \n\
	\n\
	vec3 GI = vec3(Lr,Lg,Lb); \n\
	GI*=factor; \n\
	float gamma = 2.0; \n\
	gl_FragColor = vec4(pow(GI.x,1.0-1.0/gamma),pow(GI.y,1.0-1.0/gamma),pow(GI.z,1.0-1.0/gamma),0.75); \n\
}";
