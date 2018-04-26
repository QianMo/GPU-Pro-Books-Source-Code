/*
Szymon Rusinkiewicz
Princeton University

edgeflip.cc
Optimally re-triangulate a mesh by doing edge flips
*/

#include <stdio.h>
#include "TriMesh.h"
#include "TriMesh_algo.h"
#include <utility>
#include <queue>
using std::pair;
using std::make_pair;
using std::priority_queue;

typedef pair<int,int> TriMeshEdge; // (face, edge) pair
typedef pair<float, TriMeshEdge> TriMeshEdgeWithBenefit;


// Cosine of the maximum angle in triangle (v1,v2,v3)
static float cosmaxangle(const point &v1, const point &v2, const point &v3)
{
	float a = dist(v2,v3);
	float b = dist(v3,v1);
	float c = dist(v1,v2);
	float A = a * (b*b + c*c - a*a);
	float B = b * (c*c + a*a - b*b);
	float C = c * (a*a + b*b - c*c);
	return 0.5f * min(min(A,B),C) / (a*b*c); // min cosine == max angle
}


/*
     v1         v4                      v1         v4
       +-------+                          +-------+
        \  .    \            ?             \     . \
         \   .   \          --->            \   .   \
          \     . \                          \ .     \
           +-------+                          +-------+
         v2         v3                      v2         v3

   Given a pair of triangles (v1,v2,v3) and (v1,v3,v4), what is the benefit
   of re-triangulating as (v1,v2,v4) and (v2,v3,v4)?
   In the current incarnation, makes sure we don't create a kink
   (a dihedral angle smaller than 90 degrees), then tries to minimize the
   maximum angle (which seems to work better than minimizing length of the
   diagonal and maximizing minimum angle (Delaunay)).  Note that, unlike
   the Delaunay case, this isn't guaranteed to converge to the global optimum.
*/
static float flip_benefit(const point &v1, const point &v2,
			  const point &v3, const point &v4)
{
	vec n124 = (v4 - v2) CROSS (v1 - v2);
	vec n234 = (v3 - v2) CROSS (v4 - v2);
	if ((n124 DOT n234) <= 0.0f)
		return 0.0f;

	return max(-cosmaxangle(v1,v2,v3), -cosmaxangle(v1,v3,v4)) -
	       max(-cosmaxangle(v1,v2,v4), -cosmaxangle(v2,v3,v4));
}


// Given a mesh edge defined as a (face, whichedge) pair, figure out whether
// it is possible and desirable to do an edge flip.  This does some sanity
// checks, figures out the four vertices involved, then calls the above
// function to actually compute the benefit.
static float flip_benefit(const TriMesh *mesh, int f, int e)
{
	int ae = mesh->across_edge[f][e];
	if (ae < 0)
		return 0;

	int v2 = mesh->faces[f][e];
	int v3 = mesh->faces[f][(e+1)%3];
	int v1 = mesh->faces[f][(e+2)%3];
	int faei3 = mesh->faces[ae].indexof(v3);
	if (mesh->across_edge[ae][(faei3+1)%3] != f) {
		fprintf(stderr, "AAAAAAARGH!\n");
		fprintf(stderr, "f = %d: %d %d %d\n", f, v1, v2, v3);
		fprintf(stderr, "ae = %d\n", ae);
		fprintf(stderr, "aef = %d %d %d\n", mesh->faces[ae][0], mesh->faces[ae][1], mesh->faces[ae][2]);
		fprintf(stderr, "faei3 = %d\n", faei3);
		fprintf(stderr, "aeae = %d %d %d\n", mesh->across_edge[ae][0], mesh->across_edge[ae][1], mesh->across_edge[ae][2]);
		return 0;	// Topological weirdness...
	}
	int v4 = mesh->faces[ae][(faei3+1)%3];
	if (v2 == v4)
		return 0;
	return flip_benefit(mesh->vertices[v1], mesh->vertices[v2],
			    mesh->vertices[v3], mesh->vertices[v4]);
}


// Do an edge flip of edge e on face f, updating the across_edge map.
// Returns the other face involved in the flip.
static int edge_flip(TriMesh *mesh, int f1, int e)
{
	int f2 = mesh->across_edge[f1][e];

	int v2 = mesh->faces[f1][e];
	int v3 = mesh->faces[f1][(e+1)%3];
	int f2i3 = mesh->faces[f2].indexof(v3);
	int v4 = mesh->faces[f2][(f2i3+1)%3];

	int ae14 = mesh->across_edge[f2][f2i3];
	int ae23 = mesh->across_edge[f1][(e+2)%3];

	mesh->faces[f1][(e+1)%3] = v4;
	mesh->faces[f2][(f2i3+2)%3] = v2;
	mesh->across_edge[f1][e] = ae14;
	mesh->across_edge[f1][(e+2)%3] = f2;
	mesh->across_edge[f2][f2i3] = f1;
	mesh->across_edge[f2][(f2i3+1)%3] = ae23;

	int ae14iv4 = (ae14 >= 0) ? mesh->faces[ae14].indexof(v4) : -1;
	if (ae14iv4 >= 0 && mesh->across_edge[ae14][(ae14iv4+1)%3] == f2)
		mesh->across_edge[ae14][(ae14iv4+1)%3] = f1;
	int ae23iv2 = (ae23 >= 0) ? mesh->faces[ae23].indexof(v2) : -1;
	if (ae23iv2 >= 0 && mesh->across_edge[ae23][(ae23iv2+1)%3] == f1)
		mesh->across_edge[ae23][(ae23iv2+1)%3] = f2;

	return f2;
}


// Do as many edge flips as necessary...
void edgeflip(TriMesh *mesh)
{
	mesh->need_faces();
	mesh->tstrips.clear();
	mesh->grid.clear();
	mesh->need_across_edge();

	TriMesh::dprintf("Flipping edges... ");

	// Find edges that need to be flipped, and insert them into
	// the to-do list
	int nf = mesh->faces.size();
	priority_queue<TriMeshEdgeWithBenefit> todo;
	for (int i = 0; i < nf; i++) {
		for (int j = 0; j < 3; j++) {
			float b = flip_benefit(mesh, i, j);
			if (b > 0.0f)
				todo.push(make_pair(b, make_pair(i, j)));
		}
	}


	// Process things in order of decreasing benefit
	while (!todo.empty()) {
		int f = todo.top().second.first;
		int e = todo.top().second.second;
		todo.pop();
		// Re-check in case the mesh has changed under us
		if (flip_benefit(mesh, f, e) <= 0.0f)
			continue;
		// OK, do the edge flip
		int f2 = edge_flip(mesh, f, e);
		// Insert new edges into queue, if necessary
		for (int j = 0; j < 3; j++) {
			float b = flip_benefit(mesh, f, j);
			if (b > 0.0f)
				todo.push(make_pair(b, make_pair(f, j)));
		}
		for (int j = 0; j < 3; j++) {
			float b = flip_benefit(mesh, f2, j);
			if (b > 0.0f)
				todo.push(make_pair(b, make_pair(f2, j)));
		}
	}

	TriMesh::dprintf("Done.\n");
}

