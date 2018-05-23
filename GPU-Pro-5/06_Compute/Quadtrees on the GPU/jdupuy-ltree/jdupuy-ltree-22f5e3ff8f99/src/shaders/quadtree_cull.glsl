uniform float u_scene_size;          // Terrain size

layout(std140, binding = 0)
uniform ViewFrustumPlanes {
	vec4 u_planes[6];         // frustum planes of the camera
};

layout(binding = 0, offset = 4)
uniform atomic_uint primCount; // prim counter

#ifdef VERTEX_SHADER
layout(location = 0) in uvec4 i_data;
layout(location = 0) out uvec4 o_data;

void main() {
	o_data = i_data;
}
#endif //VERTEX_SHADER


#ifdef GEOMETRY_SHADER
layout(points) in;
layout(points, max_vertices = 1) out;

layout(location = 0) in uvec4 i_data[];
layout(location = 0) out vec4 o_data;

vec3 nvertex(vec3 m, vec3 M, vec3 n) {
	bvec3 s = greaterThanEqual(n, vec3(0));
	return mix(m, M, s);
}

void main() {
	// get position
	vec2 node;
	float node_size;
	lt_cell_2_30(i_data[0].xy, node, node_size);

	// compute aabb
#if TERRAIN_RENDERER
	node = (node - 0.5) * u_scene_size;
	node_size*= u_scene_size;
	vec3 node_min = vec3(node.x, 0, node.y); // include terrain bounds
	vec3 node_max = vec3(node.x+node_size, 6553.5, node.y+node_size);
#elif PARAMETRIC_RENDERER
	vec3 p1, p2, p3, p4, t1, t2, t3;
//	ps_torus(node                     , p1, t1, t2, t3);
//	ps_torus(node + vec2(node_size, 0), p2, t1, t2, t3);
//	ps_torus(node + vec2(0, node_size), p3, t1, t2, t3);
//	ps_torus(node + vec2(node_size)   , p4, t1, t2, t3);
//	ps_sphere(node                     , p1, t1, t2, t3);
//	ps_sphere(node + vec2(node_size, 0), p2, t1, t2, t3);
//	ps_sphere(node + vec2(0, node_size), p3, t1, t2, t3);
//	ps_sphere(node + vec2(node_size)   , p4, t1, t2, t3);
	ps_trefoil_knot(node                     , p1, t1, t2, t3);
	ps_trefoil_knot(node + vec2(node_size, 0), p2, t1, t2, t3);
	ps_trefoil_knot(node + vec2(0, node_size), p3, t1, t2, t3);
	ps_trefoil_knot(node + vec2(node_size)   , p4, t1, t2, t3);
	vec3 node_min = min(min(p1,p2), min(p3,p4)) * u_scene_size;
	vec3 node_max = max(max(p1,p2), max(p3,p4)) * u_scene_size;
#else
	node = (node - 0.5) * u_scene_size;
	node_size*= u_scene_size;
	vec3 node_min = vec3(node.x, 0, node.y); // include terrain bounds
	vec3 node_max = vec3(node.x+node_size, 0, node.y+node_size);
#endif

	// cull http://www.lighthouse3d.com/tutorials/view-frustum-culling/geometric-approach-testing-boxes-ii/
	int i=0;
	float a=1;
	vec3 n;
	for(; i<6 && a>=0.0; ++i) {
		// compute negative vertex
		n = nvertex(node_min, node_max, u_planes[i].xyz);
		a = dot(u_planes[i].xyz, n) + u_planes[i].w;
	}

	// emit if intersection or inside
	if(a >= 0.0 || true) {
//		atomicCounterIncrement(primCount); // increment primCount (segfaults ?)
		o_data.xy = node;
		o_data.z = 0;
		o_data.w = node_size;
		EmitVertex();
		EndPrimitive();
	}
}
#endif //GEOMETRY_SHADER

