#ifndef SQRT_2
#	define SQRT_2 1.414213561
#endif
#ifndef PI
#	define PI 3.14159265
#endif
uniform mat4 u_mvp;
uniform vec3 u_eye_pos;

uniform float u_terrain_size;
uniform float u_time;

uniform sampler2DArray u_albedo_sampler;

#ifdef VERTEX_SHADER
layout(location = 0) in vec4 i_transformation; // translation + scale
layout(location = 1) in vec2 i_grid; // grid data
void main() {
	vec2 p = i_grid * i_transformation.w + i_transformation.xz;
	gl_Position.xzyw = vec4(p, 0, i_transformation.w);
}
#endif


#ifdef TESS_CONTROL_SHADER
layout(std140, binding=0)
uniform ViewFrustumPlanes {
	vec4 planes[6];         // frustum planes of the camera
};

vec3 nvertex(vec3 m, vec3 M, vec3 n) {
	bvec3 s = greaterThanEqual(n,vec3(0));
	return mix(m, M, s);
}

vec3 nVertex(vec3 m, vec3 M, vec3 n) {
	bvec3 s = greaterThanEqual(n,vec3(0));
	return mix(m, M, s);
}

// frustum culling test
bool frustum_test(vec3 p0, vec3 p1, vec3 p2, vec3 p3) {
	vec3 min1 = min(p0,p1);
	vec3 min2 = min(p2,p3);
	vec3 max1 = max(p0,p1);
	vec3 max2 = max(p2,p3);
	vec3 bbMin = min(min1,min2) ;
	vec3 bbMax = max(max1,max2) ;
	bbMin.y = -90.0;
	bbMax.y = 90.0;
	float a=1.0;
	for(int i=0; i<6 && a>=0.0; ++i) {
		// compute negative vertex
		vec3 n = nVertex(bbMin, bbMax, planes[i].xyz);
		a = dot(planes[i],vec4(n,1));
	}
	return a>=0.0;
}

layout(vertices=4) out;
void main() {
	if(!frustum_test(gl_in[0].gl_Position.xyz,
	                 gl_in[1].gl_Position.xyz,
	                 gl_in[2].gl_Position.xyz,
	                 gl_in[3].gl_Position.xyz)) {
		gl_TessLevelOuter[gl_InvocationID] = 0;
		gl_TessLevelInner[gl_InvocationID%2] = 0;
		return;
	}

	// find center and radius of the patch
	vec3 center  = mix(gl_in[gl_InvocationID].gl_Position.xyz,
	                   gl_in[(gl_InvocationID+1)%4].gl_Position.xyz,
	                   0.5);
#if TERRAIN_RENDERER
	center.y = displace(u_eye_pos.xz, 9e5);
#endif
	float d = distance(u_eye_pos, center);
	float s = gl_in[0].gl_Position.w;
	float tess_level = s * 2.0 * SQRT_2 / d;

	// set tess levels
	gl_TessLevelOuter[gl_InvocationID] = pow(2.0, 2.0+ceil(log2(tess_level)));
	gl_TessLevelInner[gl_InvocationID%2] = gl_TessLevelOuter[gl_InvocationID];

	// send data
	gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
}
#endif

#ifdef TESS_EVALUATION_SHADER
layout(quads, equal_spacing, ccw) in;
layout(location=0) out vec3 o_position;
void main() {
	vec3 a = mix(gl_in[0].gl_Position.xyz,
	             gl_in[3].gl_Position.xyz,
	             gl_TessCoord.x);
	vec3 b = mix(gl_in[1].gl_Position.xyz,
	             gl_in[2].gl_Position.xyz,
	             gl_TessCoord.x);
	vec3 p = mix(b, a, gl_TessCoord.y);
#if 0
	float filter = 1e2/(1.0+distance(p, u_eye_pos));
	p.y = 1*displace(p.xz, filter);
#else
	p.y = 0*displace(p.xz);
#endif
	o_position = p;
	gl_Position = u_mvp * vec4(o_position,1);
}
#endif

#ifdef FRAGMENT_SHADER
layout(location=0) in vec3 i_position;
layout(location=0) out vec4 o_colour;

#define fadeout( f, fAverage, fFeatureSize, fWidth )\
	mix(f, fAverage, smoothstep( 0.2, 0.6, fWidth / fFeatureSize))

#define filterednoise(x, a, w) fadeout(SimplexPerlin2D(x), 0.0, a, w)

float filteredfbm8(vec2 p, float w) {
	float n = 0.0;
	float a = 1.0;
	n = a * filterednoise(p, a, w); p*=2.01; a/= 1.91;
	n+= a * filterednoise(p, a, w); p*=2.03; a/= 2.01;
	n+= a * filterednoise(p, a, w); p*=1.95; a/= 1.95;
	n+= a * filterednoise(p, a, w); p*=2.11; a/= 1.99;
	n+= a * filterednoise(p, a, w); p*=1.98; a/= 2.05;
	n+= a * filterednoise(p, a, w); p*=2.10; a/= 2.03;
	n+= a * filterednoise(p, a, w); p*=1.87; a/= 1.89;
	n+= a * filterednoise(p, a, w); p*=1.93; a/= 2.10;
	return n;
}

float fbm8(vec2 p, out float variance) {
	float a = 1.0;
	float n = 0.0;
	vec2 o;
	p*= 0.02;
	variance = 0;
	o = vec2(a, a*a) * texture2(u_noise_sampler, vec3(p,0)).rg; p*=2.11; a/= 2.11;
	n+= o.x; variance+= max(1e-5, o.y - o.x*o.x);
	o = vec2(a, a*a) * texture2(u_noise_sampler, vec3(p,0)).rg; p*=1.94; a/= 1.94;
	n+= o.x; variance+= max(1e-5, o.y - o.x*o.x);
	o = vec2(a, a*a) * texture2(u_noise_sampler, vec3(p,0)).rg; p*=2.03; a/= 2.03;
	n+= o.x; variance+= max(1e-5, o.y - o.x*o.x);
//	variance*= 1.0;

//	variance*= 10.0;
	o = vec2(a, a*a) * texture2(u_noise_sampler, vec3(p,0)).rg; p*=2.11; a/= 1.99;
	n+= o.x; variance+= max(1e-5, o.y - o.x*o.x);
	o = vec2(a, a*a) * texture2(u_noise_sampler, vec3(p,0)).rg; p*=1.98; a/= 2.05;
	n+= o.x; variance+= max(1e-5, o.y - o.x*o.x);
	o = vec2(a, a*a) * texture2(u_noise_sampler, vec3(p,0)).rg; p*=2.10; a/= 2.03;
	n+= o.x; variance+= max(1e-5, o.y - o.x*o.x);
	o = vec2(a, a*a) * texture2(u_noise_sampler, vec3(p,0)).rg; p*=1.87; a/= 1.89;
	n+= o.x; variance+= max(1e-5, o.y - o.x*o.x);
	o = vec2(a, a*a) * texture2(u_noise_sampler, vec3(p,0)).rg;
	n+= o.x; variance+= max(1e-5, o.y - o.x*o.x);
	return n.x;
}


void main() {
	vec2 st = i_position.xz;

	vec3 t1 = texture(u_albedo_sampler, vec3(i_position.xz*2.25, mod(u_time*4.0, 7.0f))).rgb;
	vec3 t2 = texture(u_albedo_sampler, vec3(i_position.xz*2.25, 8.0)).rgb;
	vec3 t3 = texture(u_albedo_sampler, vec3(i_position.xz*2.25, 9.0)).rgb;; // dry grass
	vec3 t4 = vec3(1);
	o_colour.a = 1.0;

// noise correlation
#if 1
	vec2 stats;
	stats.x = fbm8(st, stats.y);
	// filtered
	o_colour.rgb = t1 * noise_cor(vec2(-1e3, 0.2), stats);
	o_colour.rgb+= t2 * noise_cor(vec2(0.2, 1.5), stats);
	o_colour.rgb+= t3 * noise_cor(vec2(1.5, 1e3), stats);
	// nfiltered
//	o_colour.rgb = t1 * (1.0 - step(0.2, stats.x));
//	o_colour.rgb+= t2 * (1.0 - step(1.5, stats.x)) * step(0.2, stats.x);
//	o_colour.rgb+= t3 * step(1.5, stats.x);
	return;
#endif

// height correlation
#if 0
	// compute shadowing term
	vec3 v = normalize(u_eye_pos - i_position).zxy;
	vec2 s;
	vec3 cov;
	vec3 n = vec3(0,1,0);
	vec3 gn = vec3(0,0,1);
	vec3 gt = vec3(1,0,0);
	vec3 gb = vec3(0,1,0);
	vec2 stats;

	stats.x = displace(st, stats.y, s, cov);
	float shadow = 1.0 + Lambda(v, gt, gb, gn, s.x, s.y, cov.x, cov.y, 0);

	o_colour.rgb = stats.xxx/4.0*0.5+0.5;
	o_colour.rgb = stats.yyy;
	// filtered
	o_colour.rgb = t1 * height_cor(vec2(-1e5, 0.2), stats, shadow);
	o_colour.rgb+= t3 * height_cor(vec2(0.2, 1e5), stats, shadow);
	// nfiltered
//	o_colour.rgb = mix(t1, t3, step(0.2, stats.x));
//	o_colour.rgb = cov.xxx;///16.0;
//	o_colour.rgb*= normalize(vec3(-s,1)).zzz;
//	o_colour.rgb = vec3(shadow - 1.0);
	return;
#endif

#if 1 // fog with distance
	float att = exp(- distance(i_position, u_eye_pos)*1e-3);
	o_colour.rgb = mix(o_colour.rgb, vec3(0.1,0.2,0.5), 1.0 - att);
#endif
}
#endif // _FRAGMENT_

