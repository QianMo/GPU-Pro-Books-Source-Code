#ifndef SQRT_2
#define SQRT_2 1.414213562
#endif

uniform vec3 u_eye_pos;
uniform float u_scene_size;

#ifdef VERTEX_SHADER
layout(location = 0) in uvec4 i_data;
layout(location = 0) out uvec4 o_data;
void main() {
	o_data = i_data;
}
#endif


#ifdef GEOMETRY_SHADER
layout(points) in;
layout(points, max_vertices = 4) out;

layout(location = 0) in uvec4 i_data[];
layout(location = 0) out uvec4 o_data;

void main() {
	// get position
	vec3 node, parent;
	float node_size;
	lt_cell_2_30(i_data[0].xy, node.xz, node_size, parent.xz);

#if PARAMETRIC_RENDERER
	vec3 p1, p2, t1, t2, t3;
//	ps_torus(node.xz + node_size * 0.5, p1, t1, t2, t3);
//	ps_torus(parent.xz + node_size, p2, t1, t2, t3);
//	ps_sphere(node.xz + node_size * 0.5, p1, t1, t2, t3);
//	ps_sphere(parent.xz + node_size, p2, t1, t2, t3);
	ps_trefoil_knot(node.xz + node_size * 0.5, p1, t1, t2, t3);
	ps_trefoil_knot(parent.xz + node_size, p2, t1, t2, t3);
	node = p1 * u_scene_size;
	parent = p2 * u_scene_size;
	node_size*= u_scene_size;
#elif TERRAIN_RENDERER
	float altitude = displace(u_eye_pos.xz / u_scene_size + 0.5, 9e5);
	node.xz-= 0.5;
	parent.xz-= 0.5;
	node_size*= u_scene_size;
	node = vec3(u_scene_size * node.xz + 0.5 * node_size, altitude).xzy;
	parent = vec3(u_scene_size * parent.xz + node_size, altitude).xzy;
#else // simple quadtree
	node.xz-= 0.5;
	parent.xz-= 0.5;
	node_size*= u_scene_size;
	node = vec3(u_scene_size * node.xz + 0.5 * node_size, 0).xzy;
	parent = vec3(u_scene_size * parent.xz + node_size, 0).xzy;
#endif

	// distance from node centers
	float dn = distance(u_eye_pos, node);
	float dp = distance(u_eye_pos, parent);

	// merge
	if(!lt_is_root_2_30(i_data[0].xy) && 8.0 * node_size < dp) {
		// make sure we generate the root node only once
		if(lt_is_upper_left_2_30(i_data[0].xy)) { 
			o_data.xy = lt_parent_2_30(i_data[0].xy);
			EmitVertex();
			EndPrimitive();
		}
	} else if(!lt_is_leaf_2_30(i_data[0].xy) && 4.0 * node_size > dn) { // split
		uvec2 children[4];
		lt_children_2_30(i_data[0].xy, children);
		o_data.xy = children[0];
		EmitVertex();
		EndPrimitive();

		o_data.xy = children[1];
		EmitVertex();
		EndPrimitive();

		o_data.xy = children[2];
		EmitVertex();
		EndPrimitive();

		o_data.xy = children[3];
		EmitVertex();
		EndPrimitive();
	} else { // keep
		o_data.xy = i_data[0].xy;
		EmitVertex();
		EndPrimitive();
	}
}
#endif

