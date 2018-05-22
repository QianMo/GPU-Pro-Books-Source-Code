/**
* Practical Elliptical Texture Filtering on the GPU
* Copyright 2010-2011 Pavlos Mavridis, All rights reserved.
*
* Version: 0.6 - 12 / 7 / 2011 (DD/MM/YY)
*/

#version 150

//========= TEXTURE FILTERING (EWA) PARAMETERS =========
#define MAX_ECCENTRICITY 16
#define NUM_PROBES 6
#define FILTER_WIDTH 1.0
#define FILTER_SHARPNESS 2.0
#define TEXELS_PER_PIXEL 1.0
#define USE_GL4 1
#define RENDER_SCENE 1
#define SPLIT_SCREEN 1
#define USE_HARDWARE_LOD 1
#define TEXEL_LIMIT 128
#define FILTER_FUNC gaussFilter

//choose a filtering mode. For the highest clarity choose one of the (2-3-4)  
#define FILTERING_MODE 2

/**
* FILTERING MODES:
* 1: Hardware
* 2: EWA
* 3: EWA 2-tex
* 4: EWA 4-tex
* 5: Approximate EWA
* 6: Approximate Spatial EWA
* 7: Approximate Temporal EWA
* 8: Display hardware mip-map selection deviation
* 9: Display annisotropy levels
*/
//=====================================================================


//========= PARAMETERS FOR THE RENDERING OF THE TUNNEL/PLANE ==========
#define RESX 900
#define RESY 900
#define SPEED 0.00
#define ZOOM 0.07
//=====================================================================


uniform int frame;
uniform float time;
uniform sampler2D tex0;

#extension GL_ARB_texture_query_lod : enable
#extension GL_EXT_gpu_shader4 : enable

#define M_PI 3.14159265358979323846


//========================= FILTER FUNCTIONS =======================
// We only use the Gaussian filter function. The other filters give 
// very similar results.

float boxFilter(float r2){
	return 1.0;
}

float gaussFilter(float r2){
	float alpha = FILTER_SHARPNESS;
	return exp(-alpha * r2);
}

float triFilter(float r2){
	float alpha = FILTER_SHARPNESS;
	float r= sqrt(r2);
	return max(0, 1.-r/alpha);
}

float sinc(float x){
	return sin(M_PI*x)/(M_PI*x);
}

float lanczosFilter(float r2){
	if (r2==0)
		return 1;
	float r= sqrt(r2);
	return sinc(r)*sinc(r/1.3);
}

//catmull-rom filter
float crFilter(float r2){
	float r = sqrt(r2);
	return (r>=2.)?.0:(r<1.)?(3.*r*r2-5.*r2+2.):(-r*r2+5.*r2-8*r+4.);
}

float quadraticFilter(float r2){
	float a = FILTER_SHARPNESS;
	return 1.0 - r2/(a*a);
}

float cubicFilter(float r2){
	float a = FILTER_SHARPNESS;
	float r = sqrt(r2);
	return 1.0 - 3*r2/(a*a) + 2*r*r2/(a*a*a);
}

//==================== EWA ( reference / 2-tex / 4-tex) ====================

/**
*	EWA filter
*	Adapted from an ANSI C implementation from Matt Pharr
*/
vec4 ewaFilter(sampler2D tex0, vec2 p0, vec2 du, vec2 dv, float lod, int psize){


	int scale = psize >> int(lod);
	vec4 foo = texture2D(tex0,p0);
	
	//don't bother with elliptical filtering if the scale is very small
	if(scale<2)
		return foo;

	p0 -=vec2(0.5,0.5)/scale;
	vec2 p = scale * p0;

	float ux = FILTER_WIDTH * du.s * scale;
    float vx = FILTER_WIDTH * du.t * scale;
    float uy = FILTER_WIDTH * dv.s * scale;
    float vy = FILTER_WIDTH * dv.t * scale;

	// compute ellipse coefficients 
    // A*x*x + B*x*y + C*y*y = F.
    float A = vx*vx+vy*vy+1;
    float B = -2*(ux*vx+uy*vy);
    float C = ux*ux+uy*uy+1;
    float F = A*C-B*B/4.;

	// Compute the ellipse's (u,v) bounding box in texture space
	float bbox_du = 2. / (-B*B+4.0*C*A) * sqrt((-B*B+4.0*C*A)*C*F);
	float bbox_dv = 2. / (-B*B+4.0*C*A) * sqrt(A*(-B*B+4.0*C*A)*F);

#if 1
	//Clamp the ellipse so that the bbox includes at most TEXEL_LIMIT texels.
	//This is necessary in order to bound the run-time, since the ellipse can be arbitrarily large
	//Note that here we are actually clamping the bbox directly instead of the ellipse.
	//Non real-time GPU renderers can skip this step.
	if(bbox_du*bbox_dv>TEXEL_LIMIT){
		float ll = sqrt(bbox_du*bbox_dv / TEXEL_LIMIT);
		bbox_du/=ll;
		bbox_dv/ll;
	}
#endif

	//the ellipse bbox			    
    int u0 = int(floor(p.s - bbox_du));
    int u1 = int(ceil (p.s + bbox_du));
    int v0 = int(floor(p.t - bbox_dv));
    int v1 = int(ceil (p.t + bbox_dv));

    // Heckbert MS thesis, p. 59; scan over the bounding box of the ellipse
    // and incrementally update the value of Ax^2+Bxy*Cy^2; when this
    // value, q, is less than F, we're inside the ellipse so we filter
    // away..
    vec4 num= vec4(0., 0., 0., 1.);
    float den = 0;
    float ddq = 2 * A;
    float U = u0 - p.s;

#if (FILTERING_MODE!=4)
	
	for (int v = v0; v <= v1; ++v) {
		float V = v - p.t;
		float dq = A*(2*U+1) + B*V;
		float q = (C*V + B*U)*V + A*U*U;
#if (FILTERING_MODE==2)
//reference implementation
		for (int u = u0; u <= u1; ++u) {
			if (q < F) 
			{
				float r2 = q / F;
				float weight = FILTER_FUNC(r2);
			
				num += weight* textureLod(tex0, vec2(u+0.5,v+0.5)/scale , int(lod));
				den += weight;
			}
			q += dq;
			dq += ddq;
		}
#else 
//FILTERING_MODE==3 / 2-tex implementation

		for (int u = u0; u <= u1; u+=2) {
			float w1 = FILTER_FUNC(q / F);
			w1 = (q < F)? w1: 0;
			q += dq;
			dq += ddq;
			float w2 = FILTER_FUNC(q / F);
			w2 = (q < F)? w2: 0;
			float offest= w2/(w1+w2);
			float weight = (w1+w2);
            if(weight>0.0)
			{
				num += weight * textureLod(tex0, vec2(u+0.5+offest, v+0.5)/scale , int(lod));
				den += weight;
            }
			q += dq;
			dq += ddq;
		}
#endif

    }

#else
//FILTERING_MODE==4 4-tex implementation
	for (int v = v0; v <= v1; v+=2) {
		float V = v - p.t;
		float dq = A*(2*U+1) + B*V;
		float q = (C*V + B*U)*V + A*U*U;
		
		float V2 = v+1 - p.t;
		float dq2 = A*(2*U+1) + B*V2;
		float q2 = (C*V2 + B*U)*V2 + A*U*U;

		for (int u = u0; u <= u1; u+=2) {
			float w1 = FILTER_FUNC(q / F);
			w1 = (q < F)? w1: 0;
			q += dq;
			dq += ddq;
			float w2 = FILTER_FUNC(q / F);
			w2 = (q < F)? w2: 0;
						
			float w3 = FILTER_FUNC(q2 / F);
			//w3 = (q2 < F)? w3: 0;
			q2 += dq2;
			dq2 += ddq;
			float w4 = FILTER_FUNC(q2 / F);
			//w4 = (q2 < F)? w4: 0;
			
			q += dq;
			dq += ddq;
			q2 += dq2;
			dq2 += ddq;
			
			float offest_v=(w3+w4)/(w1+w2+w3+w4);
			float offest_u;// = (w4+w2)/(w1+w3);
			offest_u= (w4)/(w4+w3);
			float weight =(w1+w2+w3+w4);

		//	float Error = (w1*w4-w2*w3);
			if(weight>0.1)
			{
			num += weight * textureLod(tex0, vec2(u+ offest_u+0.5, v+offest_v+0.5)/scale , int(lod));
			den += weight;
			}
		}
    }

#endif

	vec4 color = num*(1./den);
	return color;
}

//Function for mip-map lod selection
vec2 textureQueryLODEWA(sampler2D sampler, vec2 du, vec2 dv, int psize){

	int scale = psize;

	float ux = du.s * scale;
    float vx = du.t * scale;
    float uy = dv.s * scale;
    float vy = dv.t * scale;

	// compute ellipse coefficients
    // A*x*x + B*x*y + C*y*y = F.
    float A = vx*vx+vy*vy;
    float B = -2*(ux*vx+uy*vy);
    float C = ux*ux+uy*uy;
    float F = A*C-B*B/4.;
		
	A = A/F;
    B = B/F;
    C = C/F;
	
	float root=sqrt((A-C)*(A-C)+B*B);
	float majorRadius = sqrt(2./(A+C-root));
	float minorRadius = sqrt(2./(A+C+root));

	float majorLength = majorRadius;
    float minorLength = minorRadius;

	if (minorLength<0.01) minorLength=0.01;

    const float maxEccentricity = MAX_ECCENTRICITY;

    float e = majorLength / minorLength;

    if (e > maxEccentricity) {
		minorLength *= (e / maxEccentricity);
    }
	
    float lod = log2(minorLength / TEXELS_PER_PIXEL);  
	lod = clamp (lod, 0.0, log2(psize));

	return vec2(lod, e);

}

vec4 texture2DEWA(sampler2D sampler, vec2 coords){

	vec2 du = dFdx(coords);
	vec2 dv = dFdy(coords);
	
	int psize = textureSize(sampler, 0).x;
	float lod;
#if (USE_HARDWARE_LOD==1 && USE_GL4==1)
	lod = textureQueryLOD(sampler, coords).x;
#else
	lod = textureQueryLODEWA(sampler, du, dv, psize).x;
#endif

	return ewaFilter(sampler, coords, du, dv, lod, psize );

}

// visualizes the absolute deviation (error) in the hardware lod selection
vec4 lodError(sampler2D sampler, vec2 coords){

#if (USE_GL4==1)
	vec2 du = dFdx(coords);
	vec2 dv = dFdy(coords);
	
	int psize = textureSize(sampler, 0).x;

	float lod1 = textureQueryLOD(sampler, coords).x;
	float lod2 = textureQueryLODEWA(sampler, du, dv, psize).x;

	//return vec4( vec3( clamp(2*(lod2-lod1),0,1) ), 1.0);
	return vec4( vec3( abs(2*(lod2-lod1)) ), 1.0);

#else
	return vec4(0,0,0,1.0);
#endif
}

vec4 map_A(float h){
    vec4 colors[3];
    colors[0] = vec4(0.,0.,1.,1);
    colors[1] = vec4(1.,1.,0.,1);
    colors[2] = vec4(1.,0.,0.,1);

	h = clamp(h, 0 ,16);
	if(h>8)
		return mix(colors[1],colors[2], (h-8)/8);
	else
		return mix(colors[0],colors[1], h/8);
}

vec4 map_B(float h){
    vec4 colors[3];
    colors[0] = vec4(1.,0.,0.,1);
    colors[1] = vec4(0.,1.,0.,1);
    colors[2] = vec4(0.,0.,1.,1);

	h = mod(h,3);
	if(h>1)
		return mix(colors[1],colors[2], h-1);
	else
		return mix(colors[0],colors[1], h);

}


//visualizes the anisotropy level of each rendered pixel
vec4 anisoLevel(sampler2D sampler, vec2 coords){

	vec2 du = dFdx(coords);
	vec2 dv = dFdy(coords);
	
	int psize = textureSize(sampler, 0).x;

	float anisso = textureQueryLODEWA(sampler, du, dv, psize).y;

	return mix(map_A(anisso), texture2D(sampler, coords), 0.4);

}

//visualizes the mip-map level of each rendered pixel
vec4 mipLevel(sampler2D sampler, vec2 coords){

#if 0
	float lod = textureQueryLOD(sampler, coords).x;
#else
	vec2 du = dFdx(coords);
	vec2 dv = dFdy(coords);
	
	int psize = textureSize(sampler, 0).x;
	float lod = textureQueryLODEWA(sampler, du, dv, psize).x;
#endif
	return mix(map_B(lod), texture2D(sampler, coords), 0.45);

}

//==================== Approximated EWA (normal / spatial / temporal) =======================

vec4 texture2DApprox(sampler2D sampler, vec2 coords){

	vec2 du = dFdx(coords);
	vec2 dv = dFdy(coords);
	
	int psize = textureSize(sampler, 0).x;

#if (FILTERING_MODE==6)
	float vlod = textureQueryLODEWA(sampler, du, dv, psize).y;

	vec4 hcolor = texture2D(sampler, coords);
	if(vlod<12)
		return hcolor;
#endif

	int scale = psize;
	scale = 1;

	vec2 p = scale * coords;

	float ux = FILTER_WIDTH * du.s * scale;
    float vx = FILTER_WIDTH * du.t * scale;
    float uy = FILTER_WIDTH * dv.s * scale;
    float vy = FILTER_WIDTH * dv.t * scale;

	// compute ellipse coefficients to bound the region: 
    // A*x*x + B*x*y + C*y*y = F.
    float A = vx*vx+vy*vy;
    float B = -2*(ux*vx+uy*vy);
    float C = ux*ux+uy*uy;
    float F = A*C-B*B/4.;

	A = A/F;
    B = B/F;
    C = C/F;

	float root = sqrt((A-C)*(A-C)+B*B);
	float majorRadius = sqrt(2./(A+C-root));
	float minorRadius = sqrt(2./(A+C+root));

	#if 0
		//adaptive selection of probes (slower)
		float fProbes = 2.*(majorRadius/(minorRadius))-1.;
		int iProbes = int(floor(fProbes + 0.5));
		if (iProbes > NUM_PROBES) iProbes = NUM_PROBES;
	#else
		int iProbes = NUM_PROBES;
	#endif

	float lineLength = 2*(majorRadius-8*minorRadius);
	if(lineLength<0) lineLength = 0;
	//lineLength *=2.0;

	float theta= atan(B,A-C);
	if (A>C) theta = theta + M_PI/2;

	float dpu = cos(theta)*lineLength/(iProbes-1);
	float dpv = sin(theta)*lineLength/(iProbes-1);

	vec4 num = texture2D(tex0, coords);
	float den = 1;
	if(lineLength==0) iProbes=0;
	
#if (FILTERING_MODE!=7)
	for(int i=1; i<iProbes/2;i++){
	#if 1
		float d =  (float(i)/2.0)*length(vec2(dpu,dpv)) /lineLength ;
		float weight = FILTER_FUNC(d);
	#else
		float weight = 1.0 ;
	#endif

		num += weight* texture2D(tex0, coords+(i*vec2(dpu,dpv))/scale);
		num += weight* texture2D(tex0, coords-(i*vec2(dpu,dpv))/scale);

		den+=weight;
		den+=weight;
	}
#else
	//only 3 probes per frame are supported for the temporal filtering
	#if 1
	if((frame&1)==1){
		num += texture2D(tex0, (p-1*vec2(dpu,dpv))/scale );
		num += texture2D(tex0, (p+2*vec2(dpu,dpv))/scale );
		den = 3;
	}
	else{
		num += texture2D(tex0, (p+1*vec2(dpu,dpv))/scale );
		num += texture2D(tex0, (p-2*vec2(dpu,dpv))/scale );
		den = 3;
	}
	#else
		//just for debuging
		num += texture2D(tex0, (p-1*vec2(dpu,dpv))/scale );
		num += texture2D(tex0, (p+2*vec2(dpu,dpv))/scale );
		num += texture2D(tex0, (p+1*vec2(dpu,dpv))/scale );
		num += texture2D(tex0, (p-2*vec2(dpu,dpv))/scale );
		den = 5;
	#endif
#endif

#if (FILTERING_MODE==6)
	vec4 scolor = (1./den) * num;
	return mix(hcolor,scolor, smoothstep(0,1, (vlod-8.0)/13));
#else
	return (1./den) * num;
#endif

}

//==================== Texturing Wrapper Functions ==================

//drop-in replacement for glsl texture2D function
vec4 superTexture2D(sampler2D sampler, vec2 uv){
    vec4 color =  texture2D(tex0,uv);

#if (FILTERING_MODE==1)
		vec4 color2 = texture2D(tex0,uv);
#endif


#if (FILTERING_MODE==2 || FILTERING_MODE==3 || FILTERING_MODE==4 )
		vec4 color2 = texture2DEWA(tex0,uv);
#endif

#if (FILTERING_MODE==5 || FILTERING_MODE==6 || FILTERING_MODE==7 )
		vec4 color2 = texture2DApprox(tex0,uv);
#endif

#if (FILTERING_MODE==8)
		vec4 color2 = lodError(tex0,uv);
#undef SPLIT_SCREEN

#endif

#if (FILTERING_MODE==9) 
		vec4 color2 = anisoLevel(tex0,uv);
#undef SPLIT_SCREEN
#endif

#if (FILTERING_MODE==0) 
		vec4 color2 = mipLevel(tex0,uv);
#undef SPLIT_SCREEN

#endif

#if (SPLIT_SCREEN==1)
	if (abs (gl_FragCoord.x-RESX/2) <1)
		return vec4(vec3(0),1.0);

	if(gl_FragCoord.x>RESX/2)
		return color;
	else
#endif
	return color2;

}

//==================== MAIN ==================

void main(void)
{
    vec2 p = -1.0 + 2.0 * gl_FragCoord.xy / vec2(RESX,RESY);

	float time2 = 0.3 * SPEED* time;

//Two benchmark scenes adapted from Shader Toy
#if (RENDER_SCENE==1)
//TUNNEL
	float a = atan(p.y,p.x); 
	float r = ZOOM*sqrt(dot(p,p));
	vec2 uv = vec2(.75*time2+.1/r, a/3.1416);
#else
//INF_PLANES
    float an = 0.0;//0.2175;//.25;

    float x = p.x*cos(an)-p.y*sin(an);
    float y = p.x*sin(an)+p.y*cos(an);
	vec2 uv;
    uv.x = .25*x/abs(y);
    uv.y = .20*time2 + .825/abs(y);
#endif
	gl_FragColor = superTexture2D(tex0,uv);

}
