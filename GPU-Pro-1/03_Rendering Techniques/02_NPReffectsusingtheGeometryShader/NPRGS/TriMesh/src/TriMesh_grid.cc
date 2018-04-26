/*
Szymon Rusinkiewicz
Princeton University

TriMesh_grid.cc
Code for dealing with range grids
*/

#include "TriMesh.h"
#include "TriMesh_algo.h"


// Helper function - make a face with the 3 given vertices from the grid
static void mkface(TriMesh *mesh, int v1, int v2, int v3)
{
	mesh->faces.push_back(TriMesh::Face(
		mesh->grid[v1], mesh->grid[v2], mesh->grid[v3]));
}


// Triangulate a range grid
void TriMesh::triangulate_grid()
{
	dprintf("Triangulating... ");

	int nv = vertices.size();
	int ngrid = grid_width * grid_height;

	// Work around broken files that have a vertex position of (0,0,0)
	// but mark the vertex as valid, or random broken grid indices
	for (int i = 0; i < ngrid; i++) {
		if (grid[i] >= 0) {
			if (grid[i] >= nv)
				grid[i] = GRID_INVALID;
			else if (!vertices[grid[i]])
				grid[i] = GRID_INVALID;
		} else if (grid[i] < 0) {
			if (grid[i] != GRID_INVALID)
				grid[i] = GRID_INVALID;
		} else { // Some evil people like NaNs
			grid[i] = GRID_INVALID;
		}
	}

	// Count valid faces
	int ntris = 0;
	for (int j = 0; j < grid_height - 1; j++) {
		for (int i = 0; i < grid_width - 1; i++) {
			int ll = i + j * grid_width;
			int lr = ll + 1;
			int ul = ll + grid_width;
			int ur = ul + 1;
			int nvalid = (grid[ll] >= 0) + (grid[lr] >= 0) +
				     (grid[ul] >= 0) + (grid[ur] >= 0);
			if (nvalid == 4)
				ntris += 2;
			else if (nvalid == 3)
				ntris += 1;
		}
	}

	// Actually make the faces
	int old_nfaces = faces.size();
	int new_nfaces = old_nfaces + ntris;
	faces.reserve(new_nfaces);

	for (int j = 0; j < grid_height - 1; j++) {
		for (int i = 0; i < grid_width - 1; i++) {
			int ll = i + j * grid_width;
			int lr = ll + 1;
			int ul = ll + grid_width;
			int ur = ul + 1;
			int nvalid = (grid[ll] >= 0) + (grid[lr] >= 0) +
				     (grid[ul] >= 0) + (grid[ur] >= 0);
			if (nvalid < 3)
				continue;
			if (nvalid == 4) {
				// Triangulate in the direction that
				// gives the shorter diagonal
				float ll_ur = dist2(vertices[grid[ll]],
						    vertices[grid[ur]]);
				float lr_ul = dist2(vertices[grid[lr]],
						    vertices[grid[ul]]);
				if (ll_ur < lr_ul) {
					mkface(this, ll, lr, ur);
					mkface(this, ll, ur, ul);
				} else {
					mkface(this, ll, lr, ul);
					mkface(this, lr, ur, ul);
				}
				continue;
			}
			// nvalid == 3
			if (grid[ll] < 0)
				mkface(this, lr, ur, ul);
			else if (grid[lr] < 0)
				mkface(this, ll, ur, ul);
			else if (grid[ul] < 0)
				mkface(this, ll, lr, ur);
			else
				mkface(this, ll, lr, ul);
		}
	}

	dprintf("%lu faces.\n  ", ntris);
	remove_sliver_faces(this);
	dprintf("  ");
}

