/*
Szymon Rusinkiewicz
Princeton University

faceflip.cc
Flip the order of vertices in each face.  Turns the mesh inside out.
*/

#include <stdio.h>
#include "TriMesh.h"
#include "TriMesh_algo.h"


void faceflip(TriMesh *mesh)
{
	bool had_tstrips = !mesh->tstrips.empty();
	mesh->need_faces();
	mesh->tstrips.clear();

	TriMesh::dprintf("Flipping faces... ");
	int nf = mesh->faces.size();
#pragma omp parallel for
	for (int i = 0; i < nf; i++)
		swap(mesh->faces[i][0], mesh->faces[i][2]);
	TriMesh::dprintf("Done.\n");

	if (had_tstrips)
		mesh->need_tstrips();
}

