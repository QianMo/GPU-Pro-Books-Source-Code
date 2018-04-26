/*
Szymon Rusinkiewicz
Princeton University

remove.cc
Removing sets of vertices or faces from TriMeshes.
*/

#include <stdio.h>
#include "TriMesh.h"
#include "TriMesh_algo.h"


// Remove the indicated vertices from the TriMesh.
void remove_vertices(TriMesh *mesh, const vector<bool> &toremove)
{
	int nv = mesh->vertices.size();

	// Build a table that tells how the vertices will be remapped
	if (!nv)
		return;

	TriMesh::dprintf("Removing vertices... ");
	vector<int> remap_table(nv);
	int next = 0;
	for (int i = 0; i < nv; i++) {
		if (toremove[i])
			remap_table[i] = -1;
		else
			remap_table[i] = next++;
	}

	// Nothing to delete?
	if (next == nv) {
		TriMesh::dprintf("None removed.\n");
		return;
	}

	remap_verts(mesh, remap_table);

	TriMesh::dprintf("%d vertices removed... Done.\n", nv - next);
}


// Remove vertices that aren't referenced by any face
void remove_unused_vertices(TriMesh *mesh)
{
	int nv = mesh->vertices.size();
	if (!nv)
		return;

	bool had_faces = !mesh->faces.empty();
	mesh->need_faces();
	vector<bool> unused(nv, true);
	for (int i = 0; i < mesh->faces.size(); i++) {
		unused[mesh->faces[i][0]] = false;
		unused[mesh->faces[i][1]] = false;
		unused[mesh->faces[i][2]] = false;
	}
	remove_vertices(mesh, unused);
	if (!had_faces)
		mesh->faces.clear();
}


// Remove faces as indicated by toremove.  Should probably be
// followed by a call to remove_unused_vertices()
void remove_faces(TriMesh *mesh, const vector<bool> &toremove)
{
	bool had_tstrips = !mesh->tstrips.empty();
	bool had_faces = !mesh->faces.empty();
	mesh->need_faces();
	int numfaces = mesh->faces.size();
	if (!numfaces)
		return;

	mesh->tstrips.clear();
	mesh->adjacentfaces.clear();
	mesh->neighbors.clear();
	mesh->across_edge.clear();
	mesh->cornerareas.clear();
	mesh->pointareas.clear();

	TriMesh::dprintf("Removing faces... ");
	int next = 0;
	for (int i = 0; i < numfaces; i++) {
		if (toremove[i])
			continue;
		mesh->faces[next++] = mesh->faces[i];
	}
	if (next == numfaces) {
		TriMesh::dprintf("None removed.\n");
		return;
	}

	mesh->faces.erase(mesh->faces.begin() + next, mesh->faces.end());
	TriMesh::dprintf("%d faces removed... Done.\n", numfaces - next);

	if (had_tstrips)
		mesh->need_tstrips();
	if (!had_faces)
		mesh->faces.clear();

	mesh->bbox.valid = false;
	mesh->bsphere.valid = false;
}


// Remove long, skinny faces.  Should probably be followed by a
// call to remove_unused_vertices()
void remove_sliver_faces(TriMesh *mesh)
{
	mesh->need_faces();
	int numfaces = mesh->faces.size();

	const float l2thresh = sqr(4.0f * mesh->feature_size());
	const float cos2thresh = 0.85f;
	vector<bool> toremove(numfaces, false);
	for (int i = 0; i < numfaces; i++) {
		const point &v0 = mesh->vertices[mesh->faces[i][0]];
		const point &v1 = mesh->vertices[mesh->faces[i][1]];
		const point &v2 = mesh->vertices[mesh->faces[i][2]];
		float d01 = dist2(v0, v1);
		float d12 = dist2(v1, v2);
		float d20 = dist2(v2, v0);
		if (d01 < l2thresh && d12 < l2thresh && d20 < l2thresh)
			continue;
		// c2 is square of cosine of smallest angle
		float m = min(min(d01,d12),d20);
		float c2 = sqr(d01+d12+d20-2.0f*m) * m/(4.0f*d01*d12*d20);
		if (c2 < cos2thresh)
			continue;
		toremove[i] = true;
	}
	remove_faces(mesh, toremove);
}

