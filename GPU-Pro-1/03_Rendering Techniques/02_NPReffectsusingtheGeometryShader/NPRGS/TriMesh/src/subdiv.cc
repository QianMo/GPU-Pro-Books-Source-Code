/*
Szymon Rusinkiewicz
Princeton University

subdiv.cc
Perform subdivision on a mesh.
*/


#include <stdio.h>
#include "TriMesh.h"
#include "TriMesh_algo.h"


// i+1 and i-1 modulo 3
// This way of computing it tends to be faster than using %
#define NEXT(i) ((i)<2 ? (i)+1 : (i)-2)
#define PREV(i) ((i)>0 ? (i)-1 : (i)+2)


// Compute the ordinary Loop edge stencil
static point loop(TriMesh *mesh, int f1, int f2,
		  int v0, int v1, int v2, int v3)
{
	return 0.125f * (mesh->vertices[v0] + mesh->vertices[v3]) +
	       0.375f * (mesh->vertices[v1] + mesh->vertices[v2]);
}


// The point at the opposite quadrilateral corner.
// If it doesn't exist, reflect the given point across the edge.
static point opposite(TriMesh *mesh, int f, int v)
{
	int ind = mesh->faces[f].indexof(v);
	int ae = mesh->across_edge[f][ind];
	if (ae) {
		int j = mesh->faces[ae].indexof(mesh->faces[f][NEXT(ind)]);
		return mesh->vertices[mesh->faces[ae][NEXT(j)]];
	}
	return mesh->vertices[mesh->faces[f][NEXT(ind)]] +
	       mesh->vertices[mesh->faces[f][PREV(ind)]] -
	       mesh->vertices[v];
}


// Compute the butterfly stencil
static point butterfly(TriMesh *mesh, int f1, int f2,
		       int v0, int v1, int v2, int v3)
{
	point p = 0.5f * (mesh->vertices[v1] + mesh->vertices[v2]) +
		  0.125f * (mesh->vertices[v0] + mesh->vertices[v3]);
	p -= 0.0625f * (opposite(mesh, f1, v1) + opposite(mesh, f1, v2) +
			opposite(mesh, f2, v1) + opposite(mesh, f2, v2));
	return p;
}


// Compute Loop's new edge mask for an extraordinary vertex for SUBDIV_LOOP_NEW
static point new_loop_edge(TriMesh *mesh, int f1, int f2,
			   int v0, int v1, int v2, int v3)
{
	static const float wts[6][5] = {
		{ 0, 0, 0, 0, 0 },
		{ 0, 0, 0, 0, 0 },
		{ 0, 0, 0, 0, 0 },
		{ 0, 0, 0, 0, 0 },
		{ 0.3828125f, 0.125f, 0.0078125f, 0.125f, 0 },
		{ 0.3945288f, 0.1215267f, 0.01074729f, 0.01074729f, 0.1215267f }, 
	};

	int n = mesh->adjacentfaces[v1].size();
	if (n <= 3)
		return loop(mesh, f1, f2, v0, v1, v2, v3);
	point p;
	float sumwts = 0.0f;
	int f = f1;
	float s1 = 1.0f / n;
	float s2 = 2.0f * (float)M_PI * s1;
	float l = 0.375f + 0.25f * cos(s2);
	float a = (2.0f * l*l*l) / ((1.0f - l) * n);
	float b = (1.0f / l) - 1.5f;
	for (int i = 0; i < n; i++) {
		int ind = mesh->faces[f].indexof(v1);
		if (ind == -1)
			return loop(mesh, f1, f2, v0, v1, v2, v3);
		ind = NEXT(ind);
		int v = mesh->faces[f][ind];
		float wt;
		if (n < 6) {
			wt = wts[n][i];
		} else {
			float c = cos(s2 * i);
			wt = a * (1.0f + c) * sqr(b + c);
		}
		p += wt * mesh->vertices[v];
		sumwts += wt;
		f = mesh->across_edge[f][ind];
		if (f == -1)
			return loop(mesh, f1, f2, v0, v1, v2, v3);
	}
	if (f != f1)
		return loop(mesh, f1, f2, v0, v1, v2, v3);
	return p + (1.0f - sumwts) * mesh->vertices[v1];
}


// Compute Zorin's edge mask for an extraordinary vertex for
// SUBDIV_BUTTERFLY_MODIFIED
static point zorin_edge(TriMesh *mesh, int f1, int f2,
			int v0, int v1, int v2, int v3)
{
	static const float wts[6][5] = {
		{ 0, 0, 0, 0, 0 },
		{ 0, 0, 0, 0, 0 },
		{ 0, 0, 0, 0, 0 },
		{ 0.4166667f, -0.08333333f, -0.08333333f, 0, 0 },
		{ 0.375f, 0, -0.125f, 0, 0 },
		{ 0.35f, 0.0309017f, -0.0809017f, -0.0809017f, 0.0309017f },
	};

	int n = mesh->adjacentfaces[v1].size();
	if (n < 3)
		return butterfly(mesh, f1, f2, v0, v1, v2, v3);
	point p;
	float sumwts = 0.0f;
	int f = f1;
	float s1 = 1.0f / n;
	float s2 = 2.0f * (float)M_PI * s1;
	for (int i = 0; i < n; i++) {
		int ind = mesh->faces[f].indexof(v1);
		if (ind == -1)
			return butterfly(mesh, f1, f2, v0, v1, v2, v3);
		ind = NEXT(ind);
		int v = mesh->faces[f][ind];
		float wt;
		if (n < 6) {
			wt = wts[n][i];
		} else {
			float c = cos(s2 * i);
			wt = s1 * (c*c + c - 0.25f);
		}
		p += wt * mesh->vertices[v];
		sumwts += wt;
		f = mesh->across_edge[f][ind];
		if (f == -1)
			return butterfly(mesh, f1, f2, v0, v1, v2, v3);
	}
	if (f != f1)
		return butterfly(mesh, f1, f2, v0, v1, v2, v3);
	return p + (1.0f - sumwts) * mesh->vertices[v1];
}


// Average position of all boundary vertices on triangles adjacent to v
static point avg_bdy(TriMesh *mesh, int v)
{
	point p;
	int n = 0;
	const vector<int> &a = mesh->adjacentfaces[v];
	for (int i = 0; i < a.size(); i++) {
		int f = a[i];
		for (int j = 0; j < 3; j++) {
			if (mesh->across_edge[f][j] == -1) {
				p += mesh->vertices[mesh->faces[f][NEXT(j)]];
				p += mesh->vertices[mesh->faces[f][PREV(j)]];
				n += 2;
			}
		}
	}
	return p * (1.0f / n);
}


// Compute the weight on the central vertex used in updating original
// vertices in Loop subdivision
static float loop_update_alpha(int scheme, int n)
{
	if (scheme == SUBDIV_LOOP) {
		if (n == 3)	  return 0.3438f;
		else if (n == 4)  return 0.4625f;
		else if (n == 5)  return 0.5625f;
		return 0.625f;
	} else if (scheme == SUBDIV_LOOP_ORIG) {
		if (n == 3)	  return 0.4375f;
		else if (n == 4)  return 0.515625f;
		else if (n == 5)  return 0.5795339f;
		else if (n == 6)  return 0.625f;
		return 0.375f + (float)sqr(0.375f + 0.25f * cos(2.0f * M_PI / n));
	}
	// SUBDIV_LOOP_NEW
	if (n == 3)	  return 0.4375f;
	else if (n == 4)  return 0.5f;
	else if (n == 5)  return 0.545466f;
	else if (n == 6)  return 0.625f;
	float l = 0.375f + 0.25f * (float)cos(2.0f * M_PI / n);
	float beta = l * (4.0f + l * (5.0f * l - 8.0f)) / (2.0f * (1.0f - l));
	return 1.0f - beta + l * l;
}


// Insert a new vertex
static void insert_vert(TriMesh *mesh, int scheme, int f, int e)
{
	int v1 = mesh->faces[f][NEXT(e)], v2 = mesh->faces[f][PREV(e)];
	if (scheme == SUBDIV_PLANAR) {
		point p = 0.5f * (mesh->vertices[v1] +
				  mesh->vertices[v2]);
		mesh->vertices.push_back(p);
		return;
	}

	int ae = mesh->across_edge[f][e];
	if (ae == -1) {
		// Boundary
		point p = 0.5f * (mesh->vertices[v1] +
				  mesh->vertices[v2]);
		if (scheme == SUBDIV_BUTTERFLY ||
		    scheme == SUBDIV_BUTTERFLY_MODIFIED) {
			p *= 1.5f;
			p -= 0.25f * (avg_bdy(mesh, v1) + avg_bdy(mesh, v2));
		}
		mesh->vertices.push_back(p);
		return;
	}

	int v0 = mesh->faces[f][e];
	const TriMesh::Face &aef = mesh->faces[ae];
	int v3 = aef[NEXT(aef.indexof(v1))];
	point p;
	if (scheme == SUBDIV_LOOP || scheme == SUBDIV_LOOP_ORIG) {
		p = loop(mesh, f, ae, v0, v1, v2, v3);
	} else if (scheme == SUBDIV_LOOP_NEW) {
		bool e1 = (mesh->adjacentfaces[v1].size() != 6);
		bool e2 = (mesh->adjacentfaces[v2].size() != 6);
		if (e1 && e2)
			p = 0.5f * (new_loop_edge(mesh, f, ae, v0, v1, v2, v3) +
				    new_loop_edge(mesh, ae, f, v3, v2, v1, v0));
		else if (e1)
			p = new_loop_edge(mesh, f, ae, v0, v1, v2, v3);
		else if (e2)
			p = new_loop_edge(mesh, ae, f, v3, v2, v1, v0);
		else
			p = loop(mesh, f, ae, v0, v1, v2, v3);
	} else if (scheme == SUBDIV_BUTTERFLY) {
		p = butterfly(mesh, f, ae, v0, v1, v2, v3);
	} else if (scheme == SUBDIV_BUTTERFLY_MODIFIED) {
		bool e1 = (mesh->adjacentfaces[v1].size() != 6);
		bool e2 = (mesh->adjacentfaces[v2].size() != 6);
		if (e1 && e2)
			p = 0.5f * (zorin_edge(mesh, f, ae, v0, v1, v2, v3) +
				    zorin_edge(mesh, ae, f, v3, v2, v1, v0));
		else if (e1)
			p = zorin_edge(mesh, f, ae, v0, v1, v2, v3);
		else if (e2)
			p = zorin_edge(mesh, ae, f, v3, v2, v1, v0);
		else
			p = butterfly(mesh, f, ae, v0, v1, v2, v3);
	}

	mesh->vertices.push_back(p);
}


// Subdivide a mesh
void subdiv(TriMesh *mesh, int scheme /* = SUBDIV_LOOP */)
{
	bool have_col = !mesh->colors.empty();
	bool have_conf = !mesh->confidences.empty();
	mesh->flags.clear();
	mesh->normals.clear();
	mesh->pdir1.clear(); mesh->pdir2.clear();
	mesh->curv1.clear(); mesh->curv2.clear();
	mesh->dcurv.clear();
	mesh->cornerareas.clear(); mesh->pointareas.clear();
	mesh->bbox.valid = false;
	mesh->bsphere.valid = false;
	mesh->need_faces(); mesh->tstrips.clear(); mesh->grid.clear();
	mesh->neighbors.clear();
	mesh->need_across_edge(); mesh->need_adjacentfaces();


	TriMesh::dprintf("Subdividing mesh... ");

	// Introduce new vertices
	int nf = mesh->faces.size();
	vector<TriMesh::Face> newverts(nf, TriMesh::Face(-1,-1,-1));
	int old_nv = mesh->vertices.size();
	mesh->vertices.reserve(4 * old_nv);
	vector<int> newvert_count(old_nv + 3*nf);
	if (have_col)
		mesh->colors.reserve(4 * old_nv);
	if (have_conf)
		mesh->confidences.reserve(4 * old_nv);

	for (int i = 0; i < nf; i++) {
		for (int j = 0; j < 3; j++) {
			if (newverts[i][j] != -1)
				continue;
			int ae = mesh->across_edge[i][j];
			if (ae != -1) {
				if (mesh->across_edge[ae][0] == i)
					newverts[i][j] = newverts[ae][0];
				else if (mesh->across_edge[ae][1] == i)
					newverts[i][j] = newverts[ae][1];
				else if (mesh->across_edge[ae][2] == i)
					newverts[i][j] = newverts[ae][2];
			}
			if (newverts[i][j] != -1)
				continue;

			insert_vert(mesh, scheme, i, j);
			newverts[i][j] = mesh->vertices.size() - 1;
			if (ae != -1) {
				if (mesh->across_edge[ae][0] == i)
					newverts[ae][0] = newverts[i][j];
				else if (mesh->across_edge[ae][1] == i)
					newverts[ae][1] = newverts[i][j];
				else if (mesh->across_edge[ae][2] == i)
					newverts[ae][2] = newverts[i][j];
			}
			const TriMesh::Face &v = mesh->faces[i];
			if (have_col) {
				mesh->colors.push_back(0.5f *
					mesh->colors[v[NEXT(j)]] +
					mesh->colors[v[PREV(j)]]);
			}
			if (have_conf)
				mesh->confidences.push_back(0.5f *
					mesh->confidences[v[NEXT(j)]] +
					mesh->confidences[v[PREV(j)]]);
		}
	}

	// Update old vertices
	if (scheme == SUBDIV_LOOP ||
	    scheme == SUBDIV_LOOP_ORIG ||
	    scheme == SUBDIV_LOOP_NEW) {
#pragma omp parallel for
		for (int i = 0; i < old_nv; i++) {
			point bdyavg, nbdyavg;
			int nbdy = 0, nnbdy = 0;
			int naf = mesh->adjacentfaces[i].size();
			if (!naf)
				continue;
			for (int j = 0; j < naf; j++) {
				int af = mesh->adjacentfaces[i][j];
				int afi = mesh->faces[af].indexof(i);
				int n1 = NEXT(afi);
				int n2 = PREV(afi);
				if (mesh->across_edge[af][n1] == -1) {
					bdyavg += mesh->vertices[mesh->faces[af][n2]];
					nbdy++;
				} else {
					nbdyavg += mesh->vertices[mesh->faces[af][n2]];
					nnbdy++;
				}
				if (mesh->across_edge[af][n2] == -1) {
					bdyavg += mesh->vertices[mesh->faces[af][n1]];
					nbdy++;
				} else {
					nbdyavg += mesh->vertices[mesh->faces[af][n1]];
					nnbdy++;
				}
			}

			float alpha;
			point newpt;
			if (nbdy) {
				newpt = bdyavg / (float) nbdy;
				alpha = 0.75f;
			} else if (nnbdy) {
				newpt = nbdyavg / (float) nnbdy;
				alpha = loop_update_alpha(scheme, nnbdy/2);
			} else {
				continue;
			}
			mesh->vertices[i] *= alpha;
			mesh->vertices[i] += (1.0f - alpha) * newpt;
		}
	}

	// Insert new faces
	mesh->adjacentfaces.clear(); mesh->across_edge.clear();
	mesh->faces.reserve(4*nf);
	for (int i = 0; i < nf; i++) {
		TriMesh::Face &v = mesh->faces[i];
		TriMesh::Face &n = newverts[i];
		mesh->faces.push_back(TriMesh::Face(v[0], n[2], n[1]));
		mesh->faces.push_back(TriMesh::Face(v[1], n[0], n[2]));
		mesh->faces.push_back(TriMesh::Face(v[2], n[1], n[0]));
		v = n;
	}

	TriMesh::dprintf("Done.\n");
}

