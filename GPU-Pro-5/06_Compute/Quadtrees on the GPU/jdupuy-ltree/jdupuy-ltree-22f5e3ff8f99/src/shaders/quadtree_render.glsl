#ifndef SQRT_2
#	define SQRT_2 1.414213562
#endif
#ifndef PI
#	define PI 3.14159265
#endif
uniform mat4 u_mvp;
uniform vec3 u_eye_pos;

uniform float u_scene_size;
uniform vec2 u_grid_tess_factor; // Instanced grid tessellation
uniform float u_gpu_tess_factor; // GPU tess factor

#ifdef VERTEX_SHADER
layout(location = 0) in vec4 i_transformation; // translation + scale
layout(location = 1) in vec2 i_grid; // grid data
void main() {
	vec2 p = i_grid * i_transformation.w + i_transformation.xy 
	       + i_transformation.w * 0.5;
	gl_Position = vec4(p, 0, i_transformation.w);
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
	vec3 min1 = min(p0, p1);
	vec3 min2 = min(p2, p3);
	vec3 max1 = max(p0, p1);
	vec3 max2 = max(p2, p3);
	vec3 bbMin = min(min1, min2);
	vec3 bbMax = max(max1, max2);
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

layout(vertices = 4) out;
void main() {
#if 0
	if(!frustum_test(gl_in[0].gl_Position.xyz,
	                 gl_in[1].gl_Position.xyz,
	                 gl_in[2].gl_Position.xyz,
	                 gl_in[3].gl_Position.xyz)) {
		gl_TessLevelOuter[gl_InvocationID] = 0;
		gl_TessLevelInner[gl_InvocationID%2] = 0;
		return;
	}
#endif

	// get edge data
	vec4 edge = 0.5 * gl_in[gl_InvocationID].gl_Position
	          + 0.5 * gl_in[(gl_InvocationID+1)%4].gl_Position;
	vec3 p;
#if PARAMETRIC_RENDERER
	vec3 t1, t2, t3;
//	ps_torus(edge.xy, p, t1, t2, t3);
//	ps_sphere(edge.xy, p, t1, t2, t3);
	ps_trefoil_knot(edge.xy, p, t1, t2, t3);
	edge.w*= u_scene_size;
	p*= u_scene_size;
#elif TERRAIN_RENDERER
	p = vec3(edge.xy, displace(u_eye_pos.xz / u_scene_size + 0.5, 9e5)).xzy;
#else
	p = vec3(edge.xy, 0).xzy;
#endif
	float s = 2.0 * distance(u_eye_pos, p);
	float tess_level = edge.w * 8.0 * SQRT_2 / s;

	// set tess levels
	gl_TessLevelInner[gl_InvocationID%2] = tess_level - 1.0 + pow(2.0, u_gpu_tess_factor);
	gl_TessLevelOuter[gl_InvocationID] = 
		pow(2.0, u_gpu_tess_factor + ceil(log2(tess_level)));

	// send data
	gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
}
#endif


#ifdef TESS_EVALUATION_SHADER
#if PARAMETRIC_RENDERER
layout(quads, equal_spacing, cw) in;
#else
layout(quads, equal_spacing, ccw) in;
#endif
layout(location = 0) out vec4 o_position;
void main() {
	vec2 a = mix(gl_in[0].gl_Position.xy,
	             gl_in[3].gl_Position.xy,
	             gl_TessCoord.x);
	vec2 b = mix(gl_in[1].gl_Position.xy,
	             gl_in[2].gl_Position.xy,
	             gl_TessCoord.x);
	vec2 c = mix(b, a, gl_TessCoord.y);
#if PARAMETRIC_RENDERER
	vec3 t1, t2, t3;
//	ps_torus(c, o_position, t1, t2, t3);
//	ps_sphere(c, o_position, t1, t2, t3);
	ps_trefoil_knot(c, o_position, t1, t2, t3);
	o_position*= u_scene_size;
	o_position.w = gl_in[0].gl_Position.w;
#elif TERRAIN_RENDERER
	o_position.xyz = vec3(c, displace(c / u_scene_size + 0.5, 1e5)).xzy;
	o_position.w = gl_in[0].gl_Position.w / u_scene_size;
#else
	o_position.xyz = vec3(c, 0.0).xzy;
	o_position.w = gl_in[0].gl_Position.w / u_scene_size;
#endif
	gl_Position = u_mvp * vec4(o_position.xyz, 1.0);
}
#endif

#ifdef GEOMETRY_SHADER
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
layout(location = 0) in vec4 i_position[];
layout(location = 0) out vec4 o_position;
layout(location = 1) noperspective out vec3 o_distance;

void main() {
	vec2 p0 = vec2(1024) * gl_in[0].gl_Position.xy/gl_in[0].gl_Position.w;
	vec2 p1 = vec2(1024) * gl_in[1].gl_Position.xy/gl_in[1].gl_Position.w;
	vec2 p2 = vec2(1024) * gl_in[2].gl_Position.xy/gl_in[2].gl_Position.w;
	vec2 v0 = p2 - p1;
	vec2 v1 = p2 - p0;
	vec2 v2 = p1 - p0;
	float area = abs(v1.x*v2.y - v1.y * v2.x);

	o_distance = vec3(area / length(v0), 0, 0);
	o_position = i_position[0];
	gl_Position = gl_in[0].gl_Position;
	EmitVertex();
	o_distance = vec3(0, area / length(v1), 0);
	o_position = i_position[1];
	gl_Position = gl_in[1].gl_Position;
	EmitVertex();
	o_distance = vec3(0, 0, area / length(v2));
	o_position = i_position[2];
	gl_Position = gl_in[2].gl_Position;
	EmitVertex();
	EndPrimitive();
}
#endif

#ifdef FRAGMENT_SHADER
layout(location = 0) in vec4 i_position;
#ifndef GEOMETRY_SHADER_FORBIDDEN
layout(location = 1) noperspective in vec3 i_distance;
#endif
layout(location = 0) out vec4 o_colour;
void main() {
#if TERRAIN_RENDERER
	o_colour = i_position.yyyy / 6553.5;
#else
	o_colour = vec4(6.0/255.0, 40.0/255.0, 7.0/255.0, 1.0); // dark green
	o_colour = vec4(132.0/255.0, 198.0/255.0, 154.0/255.0, 1.0); // pale green
#endif

// wireframe rendering
#ifndef GEOMETRY_SHADER_FORBIDDEN
	const vec4 c1 = vec4(6.0   / 255.0, 40.0  / 255.0, 7.0   / 255.0, 1.0); // dark green
	const vec4 c2 = vec4(132.0 / 255.0, 198.0 / 255.0, 154.0 / 255.0, 1.0); // pale green
	const vec4 c3 = vec4(255.0 / 255.0, 104.0 / 255.0, 127.0 / 255.0, 1.0); // pale red
	const vec4 c4 = vec4(102.0 / 255.0, 140.0 / 255.0, 227.0 / 255.0, 1.0); // pale blue
	const vec4 c5 = vec4(0.84,0.8619,0.839,1.0);
	float nearest_edge = min(min(i_distance[0], i_distance[1]), i_distance[2]);
	float edgef = exp2(-nearest_edge * nearest_edge * 0.025);
	float lod = mod(1.0/i_position.w, 5.0);
	vec4 c = c5;
	if(lod == 1.0) c = c2;
	if(lod == 2.0) c = c3;
	if(lod == 3.0) c = c4;
//	if(1.0/i_position.w < 25.0)
//	c = c1;

	o_colour = c;
	o_colour = mix(c, c1, smoothstep(0, 1, edgef));
//	o_colour.rgb *= exp(-pow(distance(u_eye_pos, i_position.xyz)*4e-5, 2.0));
//	o_colour = 1.0-i_position.wwww;
#endif
}
#endif // _FRAGMENT_

