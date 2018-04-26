/*
Szymon Rusinkiewicz
Princeton University

TriMesh_bounding.cc
Bounding box and bounding sphere.
*/


#include <stdio.h>
#include <float.h>
#include "TriMesh.h"
#include "bsphere.h"


// Find axis-aligned bounding box of the vertices
void TriMesh::need_bbox()
{
	if (vertices.empty() || bbox.valid)
		return;

	dprintf("Computing bounding box... ");

	bbox.min = bbox.max = vertices[0];
	for (int i = 1; i < vertices.size(); i++) {
		if (vertices[i][0] < bbox.min[0])  bbox.min[0] = vertices[i][0];
		if (vertices[i][0] > bbox.max[0])  bbox.max[0] = vertices[i][0];
		if (vertices[i][1] < bbox.min[1])  bbox.min[1] = vertices[i][1];
		if (vertices[i][1] > bbox.max[1])  bbox.max[1] = vertices[i][1];
		if (vertices[i][2] < bbox.min[2])  bbox.min[2] = vertices[i][2];
		if (vertices[i][2] > bbox.max[2])  bbox.max[2] = vertices[i][2];
	}

	bbox.valid = true; 
	dprintf("Done.\n  x = %g .. %g, y = %g .. %g, z = %g .. %g\n",
		bbox.min[0], bbox.max[0], bbox.min[1],
		bbox.max[1], bbox.min[2], bbox.max[2]);
}


// Change this to #if 0 to enable a simpler (approximate) bsphere computation
// that does not use the Miniball code
#if 1

// Compute bounding sphere of the vertices.
void TriMesh::need_bsphere()
{
	if (vertices.empty() || bsphere.valid)
		return;

	dprintf("Computing bounding sphere... ");

	Miniball<3,float> mb;
	mb.check_in(vertices.begin(), vertices.end());
	mb.build();
	bsphere.center = mb.center();
	bsphere.r = sqrt(mb.squared_radius());
	bsphere.valid = true; 

	dprintf("Done.\n  center = (%g, %g, %g), radius = %g\n",
		bsphere.center[0], bsphere.center[1],
		bsphere.center[2], bsphere.r);
}

#else

// Find extreme vertex in a given direction
static int farthest_vertex_along(const TriMesh &t, const vec &dir)
{
	const vector<point> &v = t.vertices;
	int nv = v.size();

	int farthest = 0;
	float farthest_dot = v[0] DOT dir;

	for (int i = 1; i < nv; i++) {
		float my_dot = v[i] DOT dir;
		if (my_dot > farthest_dot)
			farthest = i, farthest_dot = my_dot;
	}
	return farthest;
}


// Approximate bounding sphere code based on an algorithm by Ritter
void TriMesh::need_bsphere()
{
	if (vertices.empty() || bsphere.valid)
		return;

	need_bbox();
	dprintf("Computing bounding sphere... ");

	point best_min, best_max;
	vector<vec> dirs;
	dirs.push_back(vec(1,0,0));
	dirs.push_back(vec(0,1,0));
	dirs.push_back(vec(0,0,1));
	dirs.push_back(vec(1,1,1));
	dirs.push_back(vec(1,-1,1));
	dirs.push_back(vec(1,-1,-1));
	dirs.push_back(vec(1,1,-1));
	for (int i = 0; i < dirs.size(); i++) {
		point p1 = vertices[farthest_vertex_along(*this, -dirs[i])];
		point p2 = vertices[farthest_vertex_along(*this,  dirs[i])];
		if (dist2(p1, p2) > dist2(best_min, best_max)) {
			best_min = p1;
			best_max = p2;
		}
	}
	bsphere.center = 0.5f * (best_min + best_max);
	bsphere.r = 0.5f * dist(best_min, best_max);
	float r2 = sqr(bsphere.r);

	// Expand bsphere to contain all points
	for (int i = 0; i < vertices.size(); i++) {
		float d2 = dist2(vertices[i], bsphere.center);
		if (d2 <= r2)
			continue;
		float d = sqrt(d2);
		bsphere.r = 0.5f * (bsphere.r + d);
		r2 = sqr(bsphere.r);
		bsphere.center -= vertices[i];
		bsphere.center *= bsphere.r / d;
		bsphere.center += vertices[i];
	}

	bsphere.valid = true; 
	dprintf("Done.\n  center = (%g, %g, %g), radius = %g\n",
		bsphere.center[0], bsphere.center[1],
		bsphere.center[2], bsphere.r);
}

#endif

