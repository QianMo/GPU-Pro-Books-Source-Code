#include "shaders/DeferredRendererShader_MRT.h"
#include "DeferredRenderer.h"

DRShaderMRT::DRShaderMRT()
{
	initialized = false;
}

DRShaderMRT::~DRShaderMRT()
{

}

void DRShaderMRT::start()
{
	DRShader::start();
	
/*
	static int ewaswitch = 0;
	ewaswitch=(ewaswitch+1)%60;
	int ewaon=ewaswitch>30?1:0;
	if (ewaon)
		printf("\b+");
	else
		printf("\b-");
    shader->setUniform1i(0,ewaon,uniform_ewa);		
*/
	shader->setUniform1i(0,0,uniform_ewa);		
	shader->setUniform1i(0,0,uniform_texture1);		
	shader->setUniform1i(0,1,uniform_texture2);
	shader->setUniform1i(0,2,uniform_bump);
	shader->setUniform1i(0,3,uniform_specular);
	shader->setUniform1i(0,4,uniform_emission);
	shader->setUniform1i(0,5,uniform_noise);
	float ambient [3];
	renderer->getAmbient(ambient, ambient+1, ambient+2);
	shader->setUniform3f(0,ambient[0],ambient[1],ambient[2],uniform_ambient);
}

bool DRShaderMRT::init(class DeferredRenderer* _renderer)
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
	strcpy(shader_text_frag,DRSH_Fragment_Header);
	strcat(shader_text_frag,DRSH_Fragment_Color);
	strcat(shader_text_frag,DRSH_Fragment_Normal);
	strcat(shader_text_frag,DRSH_Fragment_Specular);
    strcat(shader_text_frag,DRSH_Fragment_Lighting);
    strcat(shader_text_frag,DRSH_Fragment_Footer);

	shader = shader_manager.loadfromMemory ("Common Vertex", shader_text_vert, "MRT Construction", shader_text_frag);
	if (!shader)
	{
		EAZD_TRACE ("DeferredRenderer::buildShaders() : ERROR - Problem compiling MRT shaders.");
		free(shader_text_frag);
		return false;
	}
	else
	{
		uniform_texture1 = shader->GetUniformLocation("texture1");
		uniform_texture2 = shader->GetUniformLocation("texture2");
		uniform_specular = shader->GetUniformLocation("specular");
		uniform_bump     = shader->GetUniformLocation("bump");
		uniform_ewa      = shader->GetUniformLocation("ewa");
		uniform_emission = shader->GetUniformLocation("emission");
		uniform_noise    = shader->GetUniformLocation("noise");
		uniform_ambient  = shader->GetUniformLocation("ambient");
		shader->BindAttribLocation(1,"tangent");
	}

	free(shader_text_frag);
	initialized = true;
	return true;

}

//----------------- Shader text ----------------------------

char DRShaderMRT::DRSH_Vertex[] = "\n\
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

char DRShaderMRT::DRSH_Fragment_Header[] = "\n\
#version 130 \n\
#extension GL_ARB_texture_query_lod : enable \n\
#define EWA_ON 1 \n\
#ifdef EWA_ON \n\
#define MAX_ECCENTRICITY 32 \n\
#define NUM_PROBES 6 \n\
#define FILTER_WIDTH 1.0 \n\
#define FILTER_SHARPNESS 2.0 \n\
#define TEXELS_PER_PIXEL 1.0 \n\
#define FILTERING_MODE 2 \n\
#define M_PI 3.1415926536 \n\
#define USE_HARDWARE_LOD 0 \n\
\n\
#define TEXEL_WATCHDOG 500 \n\
\n\
/** \n\
* FILTERING MODES: \n\
* 1: Hardware \n\
* 2: EWA \n\
* 3: EWA 2-tex \n\
* 4: EWA 4-tex \n\
* 5: Approximate EWA \n\
* 6: Approximate Spatial EWA \n\
* 7: Approximate Temporal EWA \n\
* 8: Display hardware mip-map selection deviation \n\
* 9: Display annisotropy levels \n\
*/ \n\
#define FILTER_FUNC gaussFilter \n\
float gaussFilter(float r2) \n\
{ \n\
	float alpha = FILTER_SHARPNESS; \n\
	return exp(-alpha * r2); \n\
} \n\
\n\
//==================== EWA ==================== \n\
//==================== EWA ( reference / 2-tex / 4-tex) ==================== \n\
vec4 ewaFilter(sampler2D tex0, vec2 p0, vec2 du, vec2 dv, float lod, int psize) \n\
{ \n\
	int scale = psize >> int(lod); \n\
	vec4 foo = texture2D(tex0,p0); \n\
	if(scale<2) \n\
		return foo; \n\
	p0 -=vec2(0.5,0.5)/scale; \n\
	vec2 p = scale * p0; \n\
 \n\
	float ux = FILTER_WIDTH * du.s * scale; \n\
    float vx = FILTER_WIDTH * du.t * scale; \n\
    float uy = FILTER_WIDTH * dv.s * scale; \n\
    float vy = FILTER_WIDTH * dv.t * scale; \n\
 \n\
	// compute ellipse coefficients  \n\
    // A*x*x + B*x*y + C*y*y = F. \n\
    float A = vx*vx+vy*vy+1; \n\
    float B = -2*(ux*vx+uy*vy); \n\
    float C = ux*ux+uy*uy+1; \n\
    float F = A*C-B*B/4.; \n\
 \n\
	// Compute the ellipse's (u,v) bounding box in texture space \n\
    int u0 = int(floor(p.s - 2. / (-B*B+4.0*C*A) * sqrt((-B*B+4.0*C*A)*C*F))); \n\
    int u1 = int(ceil (p.s + 2. / (-B*B+4.0*C*A) * sqrt((-B*B+4.0*C*A)*C*F))); \n\
    int v0 = int(floor(p.t - 2. / (-B*B+4.0*C*A) * sqrt(A*(-B*B+4.0*C*A)*F))); \n\
    int v1 = int(ceil (p.t + 2. / (-B*B+4.0*C*A) * sqrt(A*(-B*B+4.0*C*A)*F))); \n\
 \n\
    // Heckbert MS thesis, p. 59; scan over the bounding box of the ellipse \n\
    // and incrementally update the value of Ax^2+Bxy*Cy^2; when this \n\
    // value, q, is less than F, we're inside the ellipse so we filter \n\
    // away.. \n\
    vec4 num= vec4(0., 0., 0., 0.); \n\
    float den = 0; \n\
    float ddq = 2 * A; \n\
    float U = u0 - p.s; \n\
 \n\
 \n\
#if (TEXEL_WATCHDOG!=0) \n\
	int debug_counter=0; \n\
#endif \n\
 \n\
#if (FILTERING_MODE!=4) \n\
	 \n\
	for (int v = v0; v <= v1; ++v)  \n\
{ \n\
		float V = v - p.t; \n\
		float dq = A*(2*U+1) + B*V; \n\
		float q = (C*V + B*U)*V + A*U*U; \n\
#if (FILTERING_MODE==2) \n\
//reference implementation \n\
		for (int u = u0; u <= u1; ++u)  \n\
{ \n\
 \n\
#if (TEXEL_WATCHDOG!=0) \n\
			debug_counter++; \n\
			if(debug_counter>TEXEL_WATCHDOG) \n\
				return foo; \n\
#endif \n\
			if (q < F)  \n\
			{ \n\
				float r2 = q / F; \n\
				float weight = FILTER_FUNC(r2); \n\
			 \n\
				//num += weight* texelFetch(tex0, ivec2(u,v), int(lod)); \n\
				num += weight* textureLod(tex0, vec2(u+0.5,v+0.5)/scale , int(lod)); \n\
				//num += weight* texture2DLod(tex0, vec2(u,v)/scale , int(lod)); \n\
				den += weight; \n\
			} \n\
			q += dq; \n\
			dq += ddq; \n\
		} \n\
#else  \n\
//FILTERING_MODE==3 / 2-tex implementation \n\
 \n\
		for (int u = u0; u <= u1; u+=2) { \n\
			float w1 = FILTER_FUNC(q / F); \n\
			w1 = (q < F)? w1: 0; \n\
			q += dq; \n\
			dq += ddq; \n\
			float w2 = FILTER_FUNC(q / F); \n\
			w2 = (q < F)? w2: 0; \n\
			float offest= w2/(w1+w2); \n\
			float weight = (w1+w2); \n\
            if(weight>0.0) \n\
			{ \n\
				num += weight * textureLod(tex0, vec2(u+0.5+offest, v+0.5)/scale , int(lod)); \n\
				den += weight; \n\
            } \n\
			q += dq; \n\
			dq += ddq; \n\
		} \n\
#endif \n\
 \n\
    } \n\
 \n\
#else \n\
//FILTERING_MODE==4 4-tex implementation \n\
	for (int v = v0; v <= v1; v+=2) { \n\
		float V = v - p.t; \n\
		float dq = A*(2*U+1) + B*V; \n\
		float q = (C*V + B*U)*V + A*U*U; \n\
		 \n\
		float V2 = v+1 - p.t; \n\
		float dq2 = A*(2*U+1) + B*V2; \n\
		float q2 = (C*V2 + B*U)*V2 + A*U*U; \n\
 \n\
		for (int u = u0; u <= u1; u+=2) { \n\
			float w1 = FILTER_FUNC(q / F); \n\
			w1 = (q < F)? w1: 0; \n\
			q += dq; \n\
			dq += ddq; \n\
			float w2 = FILTER_FUNC(q / F); \n\
			w2 = (q < F)? w2: 0; \n\
						 \n\
			float w3 = FILTER_FUNC(q2 / F); \n\
			//w3 = (q2 < F)? w3: 0; \n\
			q2 += dq2; \n\
			dq2 += ddq; \n\
			float w4 = FILTER_FUNC(q2 / F); \n\
			//w4 = (q2 < F)? w4: 0; \n\
			 \n\
			q += dq; \n\
			dq += ddq; \n\
			q2 += dq2; \n\
			dq2 += ddq; \n\
			 \n\
			float offest_v=(w3+w4)/(w1+w2+w3+w4); \n\
			float offest_u;// = (w4+w2)/(w1+w3); \n\
			offest_u= (w4)/(w4+w3); \n\
			float weight =(w1+w2+w3+w4); \n\
 \n\
		//	float Error = (w1*w4-w2*w3); \n\
			if(weight>0.1) \n\
			{ \n\
			num += weight * textureLod(tex0, vec2(u+ offest_u+0.5, v+offest_v+0.5)/scale , int(lod)); \n\
			den += weight; \n\
			} \n\
		} \n\
    } \n\
 \n\
#endif \n\
 \n\
	vec4 color = num*(1./den); \n\
	return color; \n\
} \n\
 \n\
//Function for mip-map lod selection \n\
vec2 textureQueryLODEWA(sampler2D sampler, vec2 du, vec2 dv, int psize){ \n\
 \n\
	int scale = psize; \n\
 \n\
	float ux = du.s * scale; \n\
    float vx = du.t * scale; \n\
    float uy = dv.s * scale; \n\
    float vy = dv.t * scale; \n\
 \n\
	// compute ellipse coefficients \n\
    // A*x*x + B*x*y + C*y*y = F. \n\
    float A = vx*vx+vy*vy; \n\
    float B = -2*(ux*vx+uy*vy); \n\
    float C = ux*ux+uy*uy; \n\
    float F = A*C-B*B/4.; \n\
		 \n\
	A = A/F; \n\
    B = B/F; \n\
    C = C/F; \n\
	 \n\
	float root=sqrt((A-C)*(A-C)+B*B); \n\
	float majorRadius = sqrt(2./(A+C-root)); \n\
	float minorRadius = sqrt(2./(A+C+root)); \n\
 \n\
	//if (root<0.005)						//handle the corner case \n\
	//	return vec2( log2(psize), 1000); \n\
 \n\
	float majorLength = majorRadius; \n\
    float minorLength = minorRadius; \n\
 \n\
	if (minorLength<0.01) minorLength=0.01; \n\
 \n\
    const float maxEccentricity = MAX_ECCENTRICITY; \n\
 \n\
    float e = majorLength / minorLength; \n\
 \n\
    if (e > maxEccentricity) { \n\
		minorLength *= (e / maxEccentricity); \n\
    } \n\
	 \n\
    float lod = log2(minorLength / TEXELS_PER_PIXEL);   \n\
	lod = clamp (lod, 0.0, log2(psize)); \n\
 \n\
	return vec2(lod, e); \n\
 \n\
} \n\
 \n\
vec4 texture2DEWA(sampler2D sampler, vec2 coords){ \n\
 \n\
	vec2 du = dFdx(coords); \n\
	vec2 dv = dFdy(coords); \n\
	 \n\
	int psize = textureSize(sampler, 0).x; \n\
	float lod; \n\
#if (USE_HARDWARE_LOD==1 && USE_GL4==1) \n\
	lod = textureQueryLOD(sampler, coords).x; \n\
#else \n\
	lod = textureQueryLODEWA(sampler, du, dv, psize).x; \n\
#endif \n\
 \n\
	return ewaFilter(sampler, coords, du, dv, lod, psize ); \n\
 \n\
} \n\
 \n\
// visualizes the absolute deviation (error) in the hardware lod selection \n\
vec4 lodError(sampler2D sampler, vec2 coords){ \n\
 \n\
#if (USE_GL4==1) \n\
	vec2 du = dFdx(coords); \n\
	vec2 dv = dFdy(coords); \n\
	 \n\
	int psize = textureSize(sampler, 0).x; \n\
 \n\
	float lod1 = textureQueryLOD(sampler, coords).x; \n\
	float lod2 = textureQueryLODEWA(sampler, du, dv, psize).x; \n\
 \n\
	return vec4( vec3( clamp(2*(lod2-lod1),0,1) ), 1.0); \n\
 \n\
#else \n\
	return vec4(0,0,0,1.0); \n\
#endif \n\
} \n\
 \n\
vec4 map_A(float h){ \n\
    vec4 colors[3]; \n\
    colors[0] = vec4(0.,0.,1.,1); \n\
    colors[1] = vec4(1.,1.,0.,1); \n\
    colors[2] = vec4(1.,0.,0.,1); \n\
 \n\
	h = clamp(h, 0 ,16); \n\
	if(h>8) \n\
		return mix(colors[1],colors[2], (h-8)/8); \n\
	else \n\
		return mix(colors[0],colors[1], h/8); \n\
 \n\
} \n\
 \n\
vec4 map_B(float h){ \n\
    vec4 colors[3]; \n\
    colors[0] = vec4(1.,0.,0.,1); \n\
    colors[1] = vec4(0.,1.,0.,1); \n\
    colors[2] = vec4(0.,0.,1.,1); \n\
 \n\
	h = mod(h,3); \n\
	if(h>1) \n\
		return mix(colors[1],colors[2], h-1); \n\
	else \n\
		return mix(colors[0],colors[1], h); \n\
 \n\
} \n\
 \n\
 \n\
//visualizes the anisotropy level of each rendered pixel \n\
vec4 anisoLevel(sampler2D sampler, vec2 coords){ \n\
 \n\
	vec2 du = dFdx(coords); \n\
	vec2 dv = dFdy(coords); \n\
	 \n\
	int psize = textureSize(sampler, 0).x; \n\
 \n\
	float anisso = textureQueryLODEWA(sampler, du, dv, psize).y; \n\
 \n\
	return mix(map_A(anisso), texture2D(sampler, coords), 0.4); \n\
 \n\
} \n\
 \n\
//visualizes the mip-map level of each rendered pixel \n\
vec4 mipLevel(sampler2D sampler, vec2 coords){ \n\
 \n\
#if 0 \n\
	float lod = textureQueryLOD(sampler, coords).x; \n\
#else \n\
	vec2 du = dFdx(coords); \n\
	vec2 dv = dFdy(coords); \n\
	 \n\
	int psize = textureSize(sampler, 0).x; \n\
	float lod = textureQueryLODEWA(sampler, du, dv, psize).x; \n\
#endif \n\
	return mix(map_B(lod), texture2D(sampler, coords), 0.45); \n\
 \n\
} \n\
 \n\
//==================== Approximated EWA (normal / spatial / temporal) ======================= \n\
 \n\
vec4 texture2DApprox(sampler2D sampler, vec2 coords){ \n\
 \n\
	vec2 du = dFdx(coords); \n\
	vec2 dv = dFdy(coords); \n\
	 \n\
	int psize = textureSize(sampler, 0).x; \n\
 \n\
#if (FILTERING_MODE==6) \n\
	float vlod = textureQueryLODEWA(sampler, du, dv, psize).y; \n\
 \n\
	vec4 hcolor = texture2D(sampler, coords); \n\
	if(vlod<12) \n\
		return hcolor; \n\
#endif \n\
 \n\
	int scale = psize; \n\
//	scale = 1; \n\
 \n\
	vec2 p = scale * coords; \n\
 \n\
	float ux = FILTER_WIDTH * du.s * scale; \n\
    float vx = FILTER_WIDTH * du.t * scale; \n\
    float uy = FILTER_WIDTH * dv.s * scale; \n\
    float vy = FILTER_WIDTH * dv.t * scale; \n\
 \n\
	// compute ellipse coefficients to bound the region:  \n\
    // A*x*x + B*x*y + C*y*y = F. \n\
    float A = vx*vx+vy*vy; \n\
    float B = -2*(ux*vx+uy*vy); \n\
    float C = ux*ux+uy*uy; \n\
    float F = A*C-B*B/4.; \n\
 \n\
	A = A/F; \n\
    B = B/F; \n\
    C = C/F; \n\
 \n\
	float root = sqrt((A-C)*(A-C)+B*B); \n\
	float majorRadius = sqrt(2./(A+C-root)); \n\
	float minorRadius = sqrt(2./(A+C+root)); \n\
 \n\
	#if 0 \n\
		float fProbes = 2.*(majorRadius/(minorRadius))-1.; \n\
		int iProbes = int(floor(fProbes + 0.5)); \n\
		if (iProbes > NUM_PROBES) iProbes = NUM_PROBES; \n\
	#else \n\
		int iProbes = NUM_PROBES; \n\
	#endif \n\
 \n\
	float lineLength = 2*(majorRadius-8*minorRadius); \n\
	if(lineLength<0) lineLength = 0; \n\
	//lineLength *=2.0; \n\
 \n\
	float theta= atan(B,A-C); \n\
	if (A>C) theta = theta + M_PI/2; \n\
 \n\
	float dpu = cos(theta)*lineLength/(iProbes-1); \n\
	float dpv = sin(theta)*lineLength/(iProbes-1); \n\
 \n\
	vec4 num = texture2D(sampler, coords); \n\
	float den = 1; \n\
	if(lineLength==0) iProbes=0; \n\
	 \n\
#if (FILTERING_MODE!=7) \n\
	for(int i=1; i<iProbes/2;i++){ \n\
	#if 1 \n\
		float d =  (float(i)/2.0)*length(vec2(dpu,dpv)) /lineLength ; \n\
		float weight = FILTER_FUNC(d); \n\
	#else \n\
		float weight = 1.0 ; \n\
	#endif \n\
 \n\
		num += weight* texture2D(sampler, coords+(i*vec2(dpu,dpv))/scale); \n\
		num += weight* texture2D(sampler, coords-(i*vec2(dpu,dpv))/scale); \n\
 \n\
		den+=weight; \n\
		den+=weight; \n\
	} \n\
#else \n\
	//only 3 probes per frame for the temporal filtering \n\
	#if 1 \n\
	if((frame&1)==1){ \n\
		num += texture2D(sampler, (p-1*vec2(dpu,dpv))/scale ); \n\
		num += texture2D(sampler, (p+2*vec2(dpu,dpv))/scale ); \n\
		den = 3; \n\
	} \n\
	else{ \n\
		num += texture2D(sampler, (p+1*vec2(dpu,dpv))/scale ); \n\
		num += texture2D(sampler, (p-2*vec2(dpu,dpv))/scale ); \n\
		den = 3; \n\
	} \n\
	#else \n\
		//for debuging \n\
		num += texture2D(sampler, (p-1*vec2(dpu,dpv))/scale ); \n\
		num += texture2D(sampler, (p+2*vec2(dpu,dpv))/scale ); \n\
		num += texture2D(sampler, (p+1*vec2(dpu,dpv))/scale ); \n\
		num += texture2D(sampler, (p-2*vec2(dpu,dpv))/scale ); \n\
		den = 5; \n\
	#endif \n\
#endif \n\
 \n\
#if (FILTERING_MODE==6) \n\
	vec4 scolor = (1./den) * num; \n\
	return mix(hcolor,scolor, smoothstep(0,1, (vlod-8.0)/13)); \n\
#else \n\
	return (1./den) * num; \n\
#endif \n\
 \n\
} \n\
 \n\
//==================== Texturing Wrapper Function ================== \n\
 \n\
vec4 superTexture2D(sampler2D sampler, vec2 uv){ \n\
    vec4 col =  texture2D(sampler,uv); \n\
 \n\
#if (FILTERING_MODE==1) \n\
		vec4 col2 = texture2D(sampler,uv); \n\
#endif \n\
 \n\
#if (FILTERING_MODE==2 || FILTERING_MODE==3 || FILTERING_MODE==4 ) \n\
		vec4 col2 = texture2DEWA(sampler,uv); \n\
#endif \n\
 \n\
#if (FILTERING_MODE==5 || FILTERING_MODE==6 || FILTERING_MODE==7 ) \n\
		vec4 col2 = texture2DApprox(sampler,uv); \n\
#endif \n\
 \n\
#if (FILTERING_MODE==8) \n\
		vec4 col2 = lodError(sampler,uv); \n\
#undef SPLIT_SCREEN \n\
 \n\
#endif \n\
 \n\
#if (FILTERING_MODE==9)  \n\
		vec4 col2 = anisoLevel(sampler,uv); \n\
#undef SPLIT_SCREEN \n\
#endif \n\
 \n\
#if (FILTERING_MODE==0)  \n\
		vec4 col2 = mipLevel(sampler,uv); \n\
#undef SPLIT_SCREEN \n\
 \n\
#endif \n\
 \n\
#if (SPLIT_SCREEN==1) \n\
	if (abs (gl_FragCoord.x-RESX/2) <1) \n\
		return vec4(vec3(0),1.0); \n\
 \n\
	if(gl_FragCoord.x>RESX/2) \n\
		return col; \n\
	else \n\
#endif \n\
	return col2; \n\
 \n\
} \n\
\n\
varying vec3 Necs; \n\
varying vec4 Pecs; \n\
varying vec3 Tecs; \n\
varying vec3 Becs; \n\
uniform int ewa; \n\
uniform sampler2D texture1, texture2, noise, bump, emission, specular; \n\
uniform vec3 ambient; \n\
float detectTexture(in sampler2D tex) \n\
{ \n\
   vec4 test = texture2D(tex, vec2(0.5,0.5))+ \n\
               texture2D(tex, vec2(0.25,0.25)); \n\
   return (sign (test.r+test.g+test.b+test.a-0.0001)+1)/2; \n\
   //return 1.0; \n\
} \n\
\n\
void main(void) \n\
{";

char DRShaderMRT::DRSH_Fragment_Footer[] = "\n\
}";

char DRShaderMRT::DRSH_Fragment_Color[] = "\n\
float hastex1 = detectTexture(texture1); \n\
vec4 tex_color1; \n\
#ifdef EWA_ON \n\
if (ewa==1) \n\
   tex_color1 = mix(vec4(1,1,1,1),superTexture2D(texture1, gl_TexCoord[0].st),hastex1); \n\
else \n\
   tex_color1 = mix(vec4(1,1,1,1),texture2D(texture1, gl_TexCoord[0].st),hastex1); \n\
#else \n\
tex_color1 = mix(vec4(1,1,1,1),texture2D(texture1, gl_TexCoord[0].st),hastex1); \n\
#endif \n\
vec4 tex_color2 = texture2D(texture2, gl_TexCoord[1].st); \n\
tex_color2.a *= detectTexture(texture2); \n\
vec4 tex_comb = gl_FrontMaterial.diffuse*mix(tex_color1,tex_color2,tex_color2.a); \n\
float alpha_clamp = max(0.0,sign(tex_comb.a-texture2D(noise, 2*gl_TexCoord[0].st).r)); \n\
gl_FragData[0] = vec4(tex_comb.rgb,tex_comb.a); \n\
";

char DRShaderMRT::DRSH_Fragment_Normal[] = "\n\
vec3 newN = Necs;\n\
vec4 nmap = texture2D(bump, gl_TexCoord[0].st);\n\
float heigh_prev_U = texture2D(bump, gl_TexCoord[0].st-(1.0/512.0,0.0)).r;\n\
float heigh_prev_V = texture2D(bump, gl_TexCoord[0].st-(0.0,1.0/512.0)).r;\n\
newN+= -2.0*(Tecs*(nmap.r-heigh_prev_U) + Becs*(nmap.r-heigh_prev_V));\n\
normalize(newN);\n\
float em = (gl_FrontMaterial.emission.x+gl_FrontMaterial.emission.y+gl_FrontMaterial.emission.z)/3.0; \n\
em += texture2D(emission, gl_TexCoord[0].st).r;\n\
gl_FragData[1] = vec4(0.5+newN.x/2.0, 0.5+newN.y/2.0,newN.z, em); \n\
";

char DRShaderMRT::DRSH_Fragment_Specular[] = "\n\
vec4 spec_coefs = vec4(gl_FrontMaterial.specular.rgb,gl_FrontMaterial.shininess/127); \n\
spec_coefs += texture2D(specular, gl_TexCoord[0].st)*vec4(1.0,1.0,1.0,1.0/127.0); \n\
gl_FragData[2] = spec_coefs; \n\
";

char DRShaderMRT::DRSH_Fragment_Lighting[] = "\n\
gl_FragData[3] = vec4(0,0,0,1); \n\
";