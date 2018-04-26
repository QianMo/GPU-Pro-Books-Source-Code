#ifndef TRIMESH_H
#define TRIMESH_H
/*
Szymon Rusinkiewicz
Princeton University

TriMesh.h
Class for triangle meshes.
*/

#include "Vec.h"
#include "Color.h"
#include <vector>
using std::vector;


class TriMesh {
protected:
	static bool read_helper(const char *filename, TriMesh *mesh);

public:
	// Types
	struct Face {
		int v[3];

		Face() {}
		Face(const int &v0, const int &v1, const int &v2)
			{ v[0] = v0; v[1] = v1; v[2] = v2; }
		Face(const int *v_)
			{ v[0] = v_[0]; v[1] = v_[1]; v[2] = v_[2]; }
		int &operator[] (int i) { return v[i]; }
		const int &operator[] (int i) const { return v[i]; }
		operator const int * () const { return &(v[0]); }
		operator const int * () { return &(v[0]); }
		operator int * () { return &(v[0]); }
		int indexof(int v_) const
		{
			return (v[0] == v_) ? 0 :
			       (v[1] == v_) ? 1 :
			       (v[2] == v_) ? 2 : -1;
		}
	};

	struct BBox {
		point min, max;
		point center() const { return 0.5f * (min+max); }
		vec size() const { return max - min; }
		bool valid;
		BBox() : valid(false)
			{}
	};

	struct BSphere {
		point center;
		float r;
		bool valid;
		BSphere() : valid(false)
			{}
	};

	// Enums
	enum tstrip_rep { TSTRIP_LENGTH, TSTRIP_TERM };
	enum { GRID_INVALID = -1 };

	// The basics: vertices and faces
	vector<point> vertices;
	vector<Face> faces;

	// Triangle strips
	vector<int> tstrips;

	// Grid, if present
	vector<int> grid;
	int grid_width, grid_height;

	// Other per-vertex properties
	vector<Color> colors;
	vector<float> confidences;
	vector<unsigned> flags;
	unsigned flag_curr;

	// Computed per-vertex properties
	vector<vec> normals;
	vector<vec> pdir1, pdir2;
	vector<float> curv1, curv2;
	vector< Vec<4,float> > dcurv;
	vector<vec> cornerareas;
	vector<float> pointareas;

	// Bounding structures
	BBox bbox;
	BSphere bsphere;

	// Connectivity structures:
	//  For each vertex, all neighboring vertices
	vector< vector<int> > neighbors;
	//  For each vertex, all neighboring faces
	vector< vector<int> > adjacentfaces;
	//  For each face, the three faces attached to its edges
	//  (for example, across_edge[3][2] is the number of the face
	//   that's touching the edge opposite vertex 2 of face 3)
	vector<Face> across_edge;

	// Compute all this stuff...
	void need_tstrips();
	void convert_strips(tstrip_rep rep);
	void unpack_tstrips();
	void triangulate_grid();
	void need_faces()
	{
		if (!faces.empty())
			return;
		if (!tstrips.empty())
			unpack_tstrips();
		else if (!grid.empty())
			triangulate_grid();
	}
	void need_normals();
	void need_pointareas();
	void need_curvatures();
	void need_dcurv();
	void need_bbox();
	void need_bsphere();
	void need_neighbors();
	void need_adjacentfaces();
	void need_across_edge();

	// Input and output
	static TriMesh *read(const char *filename);
	void write(const char *filename);

	// Statistics
	// XXX - Add stuff here
	float feature_size();

	// Useful queries
	// XXX - Add stuff here
	bool is_bdy(int v)
	{
		if (neighbors.empty()) need_neighbors();
		if (adjacentfaces.empty()) need_adjacentfaces();
		return neighbors[v].size() != adjacentfaces[v].size();
	}
	vec trinorm(int f)
	{
		if (faces.empty()) need_faces();
		return ::trinorm(vertices[faces[f][0]], vertices[faces[f][1]],
			vertices[faces[f][2]]);
	}

	// Debugging printout, controllable by a "verbose"ness parameter
	static int verbose;
	static void set_verbose(int);
	static int dprintf(const char *format, ...);

	// Constructor
	TriMesh() : grid_width(-1), grid_height(-1), flag_curr(0)
		{}
};

#endif
