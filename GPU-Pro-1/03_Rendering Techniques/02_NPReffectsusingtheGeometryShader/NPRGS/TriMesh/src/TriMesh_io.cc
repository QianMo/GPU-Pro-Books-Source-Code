/*
Szymon Rusinkiewicz
Princeton University

TriMesh_io.cc
Input and output of triangle meshes
Can read: PLY (triangle mesh and range grid), OFF, OBJ, RAY, SM, 3DS, VVD
Can write: PLY (triangle mesh and range grid), OFF, OBJ, RAY, SM, C++
*/

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdarg.h>
#include "TriMesh.h"

#ifdef WIN32
# ifndef strncasecmp
#  define strncasecmp strnicmp
# endif
#endif

#define GET_LINE() if (!fgets(buf, 1024, f)) return false
#define COND_READ(cond, where, len) if ((cond) && !fread((void *)&(where), (len), 1, f)) return false
#define LINE_IS(text) !strncasecmp(buf, text, strlen(text))

#define BIGNUM 1.0e10


// Forward declarations
static bool read_ply(FILE *f, TriMesh *mesh);
static bool read_3ds(FILE *f, TriMesh *mesh);
static bool read_vvd(FILE *f, TriMesh *mesh);
static bool read_ray(FILE *f, TriMesh *mesh);
static bool read_obj(FILE *f, TriMesh *mesh);
static bool read_off(FILE *f, TriMesh *mesh);
static bool read_sm( FILE *f, TriMesh *mesh);

static bool read_verts_bin(FILE *f, TriMesh *mesh, bool &need_swap,
	int nverts, int vert_len, int vert_pos, int vert_norm,
	int vert_color, bool float_color, int vert_conf);
static bool slurp_verts_bin(FILE *f, TriMesh *mesh, bool need_swap,
	int nverts);
static bool read_verts_asc(FILE *f, TriMesh *mesh,
	int nverts, int vert_len, int vert_pos, int vert_norm,
	int vert_color, bool float_color, int vert_conf);
static bool read_faces_bin(FILE *f, TriMesh *mesh, bool need_swap,
	int nfaces, int face_len, int face_count, int face_idx);
static bool read_faces_asc(FILE *f, TriMesh *mesh, int nfaces,
	int face_len, int face_count, int face_idx, bool read_to_eol = false);
static bool read_strips_bin(FILE *f, TriMesh *mesh, bool need_swap);
static bool read_strips_asc(FILE *f, TriMesh *mesh);
static bool read_grid_bin(FILE *f, TriMesh *mesh, bool need_swap);
static bool read_grid_asc(FILE *f, TriMesh *mesh);

static bool ply_property(const char *buf, int &len, bool binary);
static bool we_are_little_endian();
static void check_need_swap(const point &p, bool &need_swap);
static void check_ind_range(TriMesh *mesh);
static void skip_comments(FILE *f);
static void tess(const vector<point> &verts, const vector<int> &thisface,
		 vector<TriMesh::Face> &tris);

static void write_ply_ascii(TriMesh *mesh, FILE *f,
	bool write_norm, bool float_color);
static void write_ply_binary(TriMesh *mesh, FILE *f,
	bool need_swap, bool write_norm, bool float_color);
static void write_ray(TriMesh *mesh, FILE *f);
static void write_obj(TriMesh *mesh, FILE *f);
static void write_off(TriMesh *mesh, FILE *f);
static void write_sm(TriMesh *mesh, FILE *f);
static void write_cc(TriMesh *mesh, FILE *f, const char *filename,
	bool write_norm, bool float_color);
static void write_verts_asc(TriMesh *mesh, FILE *f,
			    const char *before_vert,
			    const char *before_norm,
			    const char *before_color,
			    bool float_color,
			    const char *before_conf,
			    const char *after_line);
static void write_verts_bin(TriMesh *mesh, FILE *f, bool need_swap,
			    bool write_norm, bool write_color,
			    bool float_color, bool write_conf);
static void write_faces_asc(TriMesh *mesh, FILE *f,
			    const char *before_face, const char *after_line);
static void write_faces_bin(TriMesh *mesh, FILE *f, bool need_swap,
			    int before_face_len, const char *before_face,
			    int after_face_len, const char *after_face);
static void write_strips_asc(TriMesh *mesh, FILE *f);
static void write_strips_bin(TriMesh *mesh, FILE *f, bool need_swap);
static void write_grid_asc(TriMesh *mesh, FILE *f);
static void write_grid_bin(TriMesh *mesh, FILE *f, bool need_swap);



// Byte swap uints, ints, and floats
static inline void swap_unsigned(volatile unsigned &x)
{
	x = (x << 24u) |
	    ((x << 8u) & 0x00ff0000u) |
	    ((x >> 8u) & 0x0000ff00u) |
	    (x >> 24u);
}

static inline void swap_int(int &x)
{
	swap_unsigned(* (unsigned *)(&x));
}

static inline void swap_float(float &x)
{
	swap_unsigned(* (unsigned *)(&x));
}

static inline void swap_double(double &x)
{
	unsigned char buf[8];
	memcpy(buf, &x, 8);
	swap(buf[0], buf[7]);
	swap(buf[1], buf[6]);
	swap(buf[2], buf[5]);
	swap(buf[3], buf[4]);
	memcpy(&x, buf, 8);
}

static inline void swap_ushort(volatile unsigned short &x)
{
	x = (x << 8u) | (x >> 8u);
}

static inline void swap_short(signed short &x)
{
	swap_ushort(* (unsigned short *)(&x));
}



// unget a whole string of characters
static void pushback(const char *buf, FILE *f)
{
	const char *c = buf;
	while (*c)
		c++;
	while ((--c) >= buf)
		ungetc(*c, f);
}


// Read a TriMesh from a file.  Defined to use a helper function to make
// subclassing easier.
TriMesh *TriMesh::read(const char *filename)
{
	TriMesh *mesh = new TriMesh();

	if (read_helper(filename, mesh))
		return mesh;

	delete mesh;
	return NULL;
}


// Actually read a mesh.  Tries to figure out type of file from first
// few bytes.  Filename can be "-" for stdin
bool TriMesh::read_helper(const char *filename, TriMesh *mesh)
{
	if (!filename || *filename == '\0')
		return false;

	FILE *f = NULL;
	bool ok = false;
	int c;

	if (strcmp(filename, "-") == 0) {
		f = stdin;
		filename = "standard input";
	} else {
		f = fopen(filename, "rb");
		if (!f) {
			perror("fopen");
			goto out;
		}
	}
	dprintf("Reading %s... ", filename);

	c = fgetc(f);
	if (c == EOF) {
		fprintf(stderr, "Can't read header\n");
		goto out;
	}

	if (c == 'p') {
		// See if it's a ply file
		char buf[4];
		if (!fgets(buf, 4, f)) {
			fprintf(stderr, "Can't read header\n");
			goto out;
		}
		if (strncmp(buf, "ly", 2) == 0)
			ok = read_ply(f, mesh);
	} else if (c == 0x4d) {
		int c2 = fgetc(f);
		ungetc(c2, f);
		ungetc(c, f);
		if (c2 == 0x4d)
			ok = read_3ds(f, mesh);
	} else if (c == 'V') {
		char buf[5];
		if (!fgets(buf, 5, f)) {
			fprintf(stderr, "Can't read header\n");
			goto out;
		}
		if (strncmp(buf, "IVID", 4) == 0)
			ok = read_vvd(f, mesh);
	} else if (c == '#') {
		char buf[1024];
		fscanf(f, "%1024s", buf);
		if (LINE_IS("material") || LINE_IS("vertex") ||
		    LINE_IS("shape_")) {
			// Assume a ray file
			pushback(buf, f);
			ungetc(c, f);
			ok = read_ray(f, mesh);
		} else {
			// Assume an obj file
			ok = read_obj(f, mesh);
		}
	} else if (c == 'v' || c == 'u' || c == 'f' || c == 'g' || c == 's' || c == 'o') {
		// Assume an obj file
		ungetc(c, f);
		ok = read_obj(f, mesh);
	} else if (c == 'O') {
		// Assume an OFF file
		char buf[3];
		if (!fgets(buf, 3, f)) {
			fprintf(stderr, "Can't read header\n");
			goto out;
		}
		if (strncmp(buf, "FF", 2) == 0)
			ok = read_off(f, mesh);
	} else if (isdigit(c)) {
		// Assume an old-style sm file
		ungetc(c, f);
		ok = read_sm(f, mesh);
	} else {
		fprintf(stderr, "Unknown file type\n");
	}

out:
	if (f)
		fclose(f);
	if (!ok || (mesh->vertices.empty() && mesh->faces.empty())) {
		fprintf(stderr, "\nError reading file [%s]\n", filename);
		return false;
	}

	dprintf("Done.\n");
	check_ind_range(mesh);
	return true;
}


// Read a ply file
static bool read_ply(FILE *f, TriMesh *mesh)
{
	char buf[1024];	
	bool binary = false, need_swap = false, float_color = false;
	int result, nverts = 0, nfaces = 0, nstrips = 0, ngrid = 0;
	int vert_len = 0, vert_pos = -1, vert_norm = -1;
	int vert_color = -1, vert_conf = -1;
	int face_len = 0, face_count = -1, face_idx = -1;

	// Read file format
	GET_LINE();
	while (buf[0] && isspace(buf[0]))
		GET_LINE();
	if (LINE_IS("format binary_big_endian 1.0")) {
		binary = true;
		need_swap = we_are_little_endian();
	} else if (LINE_IS("format binary_little_endian 1.0")) {
		binary = true;
		need_swap = !we_are_little_endian();
	} else if (LINE_IS("format ascii 1.0")) {
		binary = false;
	} else {
		fprintf(stderr, "Unknown ply format or version\n");
		return false;
	}

	// Skip comments and unknown obj_info lines
	GET_LINE();
	while (LINE_IS("obj_info") || LINE_IS("comment")) {
		if (LINE_IS("obj_info num_cols"))
			sscanf(buf, "obj_info num_cols %d", &mesh->grid_width);
		if (LINE_IS("obj_info num_rows"))
			sscanf(buf, "obj_info num_rows %d", &mesh->grid_height);
		GET_LINE();
	}

	// Skip until we find vertices
	int skip1 = 0;
	while (!LINE_IS("end_header") && !LINE_IS("element vertex")) {
		char elem_name[1024];
		int nelem = 0, elem_len = 0;
		sscanf(buf, "element %s %d", elem_name, &nelem);
		GET_LINE();
		while (LINE_IS("property")) {
			if (!ply_property(buf, elem_len, binary))
				return false;
			GET_LINE();
		}
		skip1 += nelem * elem_len;
	}

	// Find number of vertices
	result = sscanf(buf, "element vertex %d\n", &nverts);
	if (result != 1) {
		fprintf(stderr, "Expected \"element vertex\"\n");
		return false;
	}

	// Parse vertex properties
	GET_LINE();
	while (LINE_IS("property")) {
		if (LINE_IS("property float x") ||
		    LINE_IS("property float32 x"))
			vert_pos = vert_len;
		if (LINE_IS("property float nx") ||
		    LINE_IS("property float32 nx"))
			vert_norm = vert_len;
		if (LINE_IS("property uchar diffuse_red") ||
		    LINE_IS("property uint8 diffuse_red") ||
		    LINE_IS("property uchar red") ||
		    LINE_IS("property uint8 red"))
			vert_color = vert_len;
		if (LINE_IS("property float diffuse_red") ||
		    LINE_IS("property float32 diffuse_red") ||
		    LINE_IS("property float red") ||
		    LINE_IS("property float32 red"))
			vert_color = vert_len, float_color = true;
		if (LINE_IS("property float confidence") ||
		    LINE_IS("property float32 confidence"))
			vert_conf = vert_len;

		if (!ply_property(buf, vert_len, binary))
			return false;

		GET_LINE();
	}

	// Skip until we find faces
	int skip2 = 0;
	while (!LINE_IS("end_header") && !LINE_IS("element face") &&
	       !LINE_IS("element tristrips") && !LINE_IS("element range_grid")) {
		char elem_name[1024];
		int nelem = 0, elem_len = 0;
		sscanf(buf, "element %s %d", elem_name, &nelem);
		GET_LINE();
		while (LINE_IS("property")) {
			if (!ply_property(buf, elem_len, binary))
				return false;
			GET_LINE();
		}
		skip2 += nelem * elem_len;
	}


	// Look for faces, tristrips, or range grid
	if (LINE_IS("element face")) {
		if (sscanf(buf, "element face %d\n", &nfaces) != 1)
			return false;
		GET_LINE();
		while (LINE_IS("property")) {
			if (LINE_IS("property list uchar int vertex_indices") ||
			    LINE_IS("property list uint8 int32 vertex_indices") ||
			    LINE_IS("property list char int vertex_indices") ||
			    LINE_IS("property list int8 int32 vertex_indices") ||
			    LINE_IS("property list uchar int vertex_index") ||
			    LINE_IS("property list uint8 int32 vertex_index") ||
			    LINE_IS("property list char int vertex_index") ||
			    LINE_IS("property list int8 int32 vertex_index")) {
				face_count = face_len;
				face_idx = face_len + 1;
				face_len += 1;
			} else if
			   (LINE_IS("property list uint int vertex_indices") ||
			    LINE_IS("property list uint32 int32 vertex_indices") ||
			    LINE_IS("property list int int vertex_indices") ||
			    LINE_IS("property list int32 int32 vertex_indices") ||
			    LINE_IS("property list uint int vertex_index") ||
			    LINE_IS("property list uint32 int32 vertex_index") ||
			    LINE_IS("property list int int vertex_index") ||
			    LINE_IS("property list int32 int32 vertex_index")) {
				face_count = face_len;
				face_idx = face_len + (binary ? 4 : 1);
				face_len += (binary ? 4 : 1);
			} else if (!ply_property(buf, face_len, binary))
				return false;
			GET_LINE();
		}
	} else if (LINE_IS("element tristrips")) {
		nstrips = 1;
		GET_LINE();
		if (!LINE_IS("property list int int vertex_indices") &&
		    !LINE_IS("property list int32 int32 vertex_indices"))
			return false;
		GET_LINE();
	} else if (LINE_IS("element range_grid")) {
		if (sscanf(buf, "element range_grid %d\n", &ngrid) != 1)
			return false;
		if (ngrid != mesh->grid_width*mesh->grid_height) {
			fprintf(stderr, "Range grid size does not equal num_rows*num_cols\n");
			return false;
		}
		GET_LINE();
		if (!LINE_IS("property list uchar int vertex_indices") &&
		    !LINE_IS("property list uint8 int32 vertex_indices") &&
		    !LINE_IS("property list char int vertex_indices") &&
		    !LINE_IS("property list int8 int32 vertex_indices"))
			return false;
		GET_LINE();
	}

	while (LINE_IS("property")) {
		if (!ply_property(buf, face_len, binary))
			return false;
		GET_LINE();
	}

	// Skip to the end of the header
	while (!LINE_IS("end_header"))
		GET_LINE();
	if (binary && buf[10] == '\r') {
		fprintf(stderr, "Warning!  Possibly corrupt file\n");
		fprintf(stderr, "     If things don't work, make sure this file was transferred in BINARY, not ASCII mode\n");
	}


	// Actually read everything in
	if (skip1) {
		if (binary)
			fseek(f, skip1, SEEK_CUR);
		else
			for (int i = 0; i < skip1; i++)
				fscanf(f, "%s", buf);
	}
	if (binary) {
		if (!read_verts_bin(f, mesh, need_swap, nverts, vert_len,
				    vert_pos, vert_norm, vert_color,
				    float_color, vert_conf))
			return false;
	} else {
		if (!read_verts_asc(f, mesh, nverts, vert_len,
				    vert_pos, vert_norm, vert_color,
				    float_color, vert_conf))
			return false;
	}

	if (skip2) {
		if (binary)
			fseek(f, skip2, SEEK_CUR);
		else
			for (int i = 0; i < skip2; i++)
				fscanf(f, "%s", buf);
	}

	if (ngrid) {
		if (binary) {
			if (!read_grid_bin(f, mesh, need_swap))
				return false;
		} else {
			if (!read_grid_asc(f, mesh))
				return false;
		}
	} else if (nstrips) {
		if (binary) {
			if (!read_strips_bin(f, mesh, need_swap))
				return false;
		} else {
			if (!read_strips_asc(f, mesh))
				return false;
		}
		mesh->convert_strips(TriMesh::TSTRIP_LENGTH);
	} else if (nfaces) {
		if (binary) {
			if (!read_faces_bin(f, mesh, need_swap, nfaces,
					    face_len, face_count, face_idx))
				return false;
		} else {
			if (!read_faces_asc(f, mesh, nfaces,
					    face_len, face_count, face_idx))
				return false;
		}
	}

	return true;
}


#define CHUNK_3DS_MAIN  0x4d4d
#define CHUNK_3DS_MODEL 0x3d3d
#define CHUNK_3DS_OBJ   0x4000
#define CHUNK_3DS_MESH  0x4100
#define CHUNK_3DS_VERT  0x4110
#define CHUNK_3DS_FACE  0x4120

// Read a 3DS file.
static bool read_3ds(FILE *f, TriMesh *mesh)
{
	bool need_swap = !we_are_little_endian();
	int mstart = 0;

	while (!feof(f)) {
		short chunkid;
		int chunklen;
		if (!fread(&chunkid, 2, 1, f) ||
		    !fread(&chunklen, 4, 1, f))
			return false;
		if (need_swap) {
			swap_short(chunkid);
			swap_int(chunklen);
		}
		//TriMesh::dprintf("Found chunk %x of length %d\n", chunkid, chunklen);
		switch (chunkid) {
			case CHUNK_3DS_MAIN:
			case CHUNK_3DS_MODEL:
				// Just recurse into this chunk
				break;

			case CHUNK_3DS_OBJ:
				// Skip name, then recurse
				while (!feof(f) && fgetc(f))
					;
				break;

			case CHUNK_3DS_MESH:
				mstart = mesh->vertices.size();
				break;

			case CHUNK_3DS_VERT: {
				unsigned short nverts;
				if (!fread(&nverts, 2, 1, f))
					return false;
				if (need_swap)
					swap_ushort(nverts);
				read_verts_bin(f, mesh, need_swap,
					       nverts, 12, 0, -1, -1, false, -1);
				break;
			}
			case CHUNK_3DS_FACE: {
				unsigned short nfaces;
				if (!fread(&nfaces, 2, 1, f))
					return false;
				if (need_swap)
					swap_ushort(nfaces);
				TriMesh::dprintf("\n  Reading %d faces... ", nfaces);
				int old_nfaces = mesh->faces.size();
				int new_nfaces = old_nfaces + nfaces;
				mesh->faces.resize(new_nfaces);
				for (int i = old_nfaces; i < new_nfaces; i++) {
					unsigned short buf[4];
					COND_READ(true, buf[0], 8);
					if (need_swap) {
						swap_ushort(buf[0]);
						swap_ushort(buf[1]);
						swap_ushort(buf[2]);
					}
					mesh->faces[i][0] = mstart + buf[0];
					mesh->faces[i][1] = mstart + buf[1];
					mesh->faces[i][2] = mstart + buf[2];
				}
				break;
			}
			default: {
				// Skip over this chunk
				fseek(f, chunklen-6, SEEK_CUR);
			}
		}
	}
	return true;
}


// Read a VVD file.
static bool read_vvd(FILE *f, TriMesh *mesh)
{
	bool need_swap = we_are_little_endian();
	const int skip = 127;
	char buf[skip];
	fread(buf, skip, 1, f);

	int nverts;
	if (fread(&nverts, 4, 1, f) != 1) {
		fprintf(stderr, "Couldn't read vertex count\n");
		return false;
	}
	if (need_swap)
		swap_int(nverts);
	mesh->vertices.resize(nverts);
	TriMesh::dprintf("\n  Reading %d vertices... ", nverts);

	for (int i = 0; i < nverts; i++) {
		double v[3];
		if (fread(&v[0], 24, 1, f) != 1) {
			fprintf(stderr, "Couldn't read vertex\n");
			return false;
		}
		if (need_swap) {
			swap_double(v[0]);
			swap_double(v[1]);
			swap_double(v[2]);
		}
		mesh->vertices[i] = point(float(v[0]), float(v[1]), float(v[2]));
	}

	int nfaces;
	if (fread(&nfaces, 4, 1, f) != 1) {
		fprintf(stderr, "Couldn't read face count\n");
		return false;
	}
	if (need_swap)
		swap_int(nfaces);
	read_faces_bin(f, mesh, need_swap, nfaces, 4, 0, 4);

	return true;
}


// Read a ray file
static bool read_ray(FILE *f, TriMesh *mesh)
{
	while (!feof(f)) {
		char buf[1024];
		buf[0] = '\0';
		if (fscanf(f, " %1024s", buf) == 0)
			return true;
		if (LINE_IS("#vertex")) {
			float x, y, z;
			if (fscanf(f, "%f %f %f", &x, &y, &z) != 3) {
				return false;
			}
			mesh->vertices.push_back(point(x,y,z));
		} else if (LINE_IS("#shape_triangle")) {
			int f1, f2, f3, m;
			if (fscanf(f, "%d %d %d %d", &m, &f1, &f2, &f3) != 4) {
				return false;
			}
			mesh->faces.push_back(TriMesh::Face(f1,f2,f3));
		}
	}
	return true;
}


// Read an obj file
static bool read_obj(FILE *f, TriMesh *mesh)
{
	vector<int> thisface;
	while (1) {
		skip_comments(f);
		if (feof(f))
			return true;
		char buf[1024];
		GET_LINE();
		if (LINE_IS("v ") || LINE_IS("v\t")) {
			float x, y, z;
			if (sscanf(buf+1, "%f %f %f", &x, &y, &z) != 3) {
				return false;
			}
			mesh->vertices.push_back(point(x,y,z));
		} else if (LINE_IS("f ") || LINE_IS("f\t") ||
			   LINE_IS("t ") || LINE_IS("t\t")) {
			thisface.clear();
			char *c = buf;
			while (1) {
				while (*c && *c != '\n' && !isspace(*c))
					c++;
				while (*c && isspace(*c))
					c++;
				int thisf;
				if (sscanf(c, " %d", &thisf) != 1)
					break;
				if (thisf < 0)
					thisf += mesh->vertices.size();
				else
					thisf--;
				thisface.push_back(thisf);
			}
			tess(mesh->vertices, thisface, mesh->faces);
		}
	}
	return true;
}


// Read an off file
static bool read_off(FILE *f, TriMesh *mesh)
{
	skip_comments(f);
	char buf[1024];
	GET_LINE();
	int nverts, nfaces, unused;
	if (sscanf(buf, "%d %d %d", &nverts, &nfaces, &unused) < 2)
		return false;
	if (!read_verts_asc(f, mesh, nverts, 3, 0, -1, -1, false, -1))
		return false;
	if (!read_faces_asc(f, mesh, nfaces, 1, 0, 1, true))
		return false;

	return true;
}


// Read an sm file
static bool read_sm(FILE *f, TriMesh *mesh)
{
	int nverts, nfaces;

	if (fscanf(f, "%d", &nverts) != 1)
		return false;

	if (!read_verts_asc(f, mesh, nverts, 3, 0, -1, -1, false, -1))
		return false;

	skip_comments(f);
	if (fscanf(f, "%d", &nfaces) != 1)
		return true;
	if (!read_faces_asc(f, mesh, nfaces, 0, -1, 0))
		return false;

	return true;
}


// Read nverts vertices from a binary file.
// vert_len = total length of a vertex record in bytes
// vert_pos, vert_norm, vert_color, vert_conf =
//   position of vertex coordinates / normals / color / confidence in record
// need_swap = swap for opposite endianness
// float_color = colors are 4-byte float * 3, vs 1-byte uchar * 3
static bool read_verts_bin(FILE *f, TriMesh *mesh, bool &need_swap,
	int nverts, int vert_len, int vert_pos, int vert_norm,
	int vert_color, bool float_color, int vert_conf)
{
	const int vert_size = 12;
	const int norm_size = 12;
	const int color_size = float_color ? 12 : 3;
	const int conf_size = 4;

	if (nverts <= 0 || vert_len < 12 || vert_pos < 0)
		return false;

	int old_nverts = mesh->vertices.size();
	int new_nverts = old_nverts + nverts;
	mesh->vertices.resize(new_nverts);

	bool have_norm = (vert_norm >= 0);
	bool have_color = (vert_color >= 0);
	bool have_conf = (vert_conf >= 0);
	if (have_norm)
		mesh->normals.resize(new_nverts);
	if (have_color)
		mesh->colors.resize(new_nverts);
	if (have_conf)
		mesh->confidences.resize(new_nverts);

	unsigned char *buf = new unsigned char[vert_len];
	COND_READ(true, buf[0], vert_len);

	int i = old_nverts;
	memcpy(&mesh->vertices[i][0], &buf[vert_pos], vert_size);
	if (have_norm)
		memcpy(&mesh->normals[i][0], &buf[vert_norm], norm_size);
	if (have_color && float_color)
		memcpy(&mesh->colors[i][0], &buf[vert_color], color_size);
	if (have_color && !float_color)
		mesh->colors[i] = Color(&buf[vert_color]);
	if (have_conf)
		memcpy(&mesh->confidences[i], &buf[vert_conf], conf_size);

	check_need_swap(mesh->vertices[i], need_swap);
	if (need_swap) {
		swap_float(mesh->vertices[i][0]);
		swap_float(mesh->vertices[i][1]);
		swap_float(mesh->vertices[i][2]);
		if (have_norm) {
			swap_float(mesh->normals[i][0]);
			swap_float(mesh->normals[i][1]);
			swap_float(mesh->normals[i][2]);
		}
		if (have_color && float_color) {
			swap_float(mesh->colors[i][0]);
			swap_float(mesh->colors[i][1]);
			swap_float(mesh->colors[i][2]);
		}
		if (have_conf)
			swap_float(mesh->confidences[i]);
	}

	TriMesh::dprintf("\n  Reading %d vertices... ", nverts);
	if (vert_len == 12 && sizeof(point) == 12 && nverts > 1)
		return slurp_verts_bin(f, mesh, need_swap, nverts);
	while (++i < new_nverts) {
		COND_READ(true, buf[0], vert_len);
		memcpy(&mesh->vertices[i][0], &buf[vert_pos], vert_size);
		if (have_norm)
			memcpy(&mesh->normals[i][0], &buf[vert_norm], norm_size);
		if (have_color && float_color)
			memcpy(&mesh->colors[i][0], &buf[vert_color], color_size);
		if (have_color && !float_color)
			mesh->colors[i] = Color(&buf[vert_color]);
		if (have_conf)
			memcpy(&mesh->confidences[i], &buf[vert_conf], conf_size);

		if (need_swap) {
			swap_float(mesh->vertices[i][0]);
			swap_float(mesh->vertices[i][1]);
			swap_float(mesh->vertices[i][2]);
			if (have_norm) {
				swap_float(mesh->normals[i][0]);
				swap_float(mesh->normals[i][1]);
				swap_float(mesh->normals[i][2]);
			}
			if (have_color && float_color) {
				swap_float(mesh->colors[i][0]);
				swap_float(mesh->colors[i][1]);
				swap_float(mesh->colors[i][2]);
			}
			if (have_conf)
				swap_float(mesh->confidences[i]);
		}
	}

	return true;
}


// Optimized reader for the simple case of just vertices w/o other properties
static bool slurp_verts_bin(FILE *f, TriMesh *mesh, bool need_swap, int nverts)
{
	int first = mesh->vertices.size() - nverts + 1;
	COND_READ(true, mesh->vertices[first][0], (nverts-1)*12);
	if (need_swap) {
	    for (int i = first; i < mesh->vertices.size(); i++) {
			swap_float(mesh->vertices[i][0]);
			swap_float(mesh->vertices[i][1]);
			swap_float(mesh->vertices[i][2]);
		}
	}
	return true;
}


// Read a bunch of vertices from an ASCII file.
// Parameters are as in read_verts_bin, but offsets are in
// (white-space-separated) words, rather than in bytes
static bool read_verts_asc(FILE *f, TriMesh *mesh,
	int nverts, int vert_len, int vert_pos, int vert_norm,
	int vert_color, bool float_color, int vert_conf)
{
	if (nverts <= 0 || vert_len < 3 || vert_pos < 0)
		return false;

	int old_nverts = mesh->vertices.size();
	int new_nverts = old_nverts + nverts;
	mesh->vertices.resize(new_nverts);
	if (vert_norm > 0)
		mesh->normals.resize(new_nverts);
	if (vert_color > 0)
		mesh->colors.resize(new_nverts);
	if (vert_conf > 0)
		mesh->confidences.resize(new_nverts);

	char buf[1024];
	skip_comments(f);
	TriMesh::dprintf("\n  Reading %d vertices... ", nverts);
	for (int i = old_nverts; i < new_nverts; i++) {
		for (int j = 0; j < vert_len; j++) {
			if (j == vert_pos) {
				if (fscanf(f, "%f %f %f",
					      &mesh->vertices[i][0],
					      &mesh->vertices[i][1],
					      &mesh->vertices[i][2]) != 3)
					return false;
				j += 2;
			} else if (j == vert_norm) {
				if (fscanf(f, "%f %f %f",
					      &mesh->normals[i][0],
					      &mesh->normals[i][1],
					      &mesh->normals[i][2]) != 3)
					return false;
				j += 2;
			} else if (j == vert_color && float_color) {
				float r, g, b;
				if (fscanf(f, "%f %f %f", &r, &g, &b) != 3)
					return false;
				mesh->colors[i] = Color(r,g,b);
				j += 2;
			} else if (j == vert_color && !float_color) {
				int r, g, b;
				if (fscanf(f, "%d %d %d", &r, &g, &b) != 3)
					return false;
				mesh->colors[i] = Color(r,g,b);
				j += 2;
			} else if (j == vert_conf) {
				if (fscanf(f, "%f", &mesh->confidences[i]) != 1)
					return false;
			} else {
				fscanf(f, " %1024s", buf);
			}
		}
	}

	return true;
}


// Read nfaces faces from a binary file.
// face_len = total length of face record, *not counting the indices*
//  (Yes, this is bizarre, but there is potentially a variable # of indices...)
// face_count = offset within record of the count of indices in this face
//  (If this is -1, does not read a count and assumes triangles)
// face_idx = offset within record of the indices themselves
static bool read_faces_bin(FILE *f, TriMesh *mesh, bool need_swap,
	int nfaces, int face_len, int face_count, int face_idx)
{
	if (nfaces < 0 || face_idx < 0)
		return false;

	if (nfaces == 0)
		return true;

	TriMesh::dprintf("\n  Reading %d faces... ", nfaces);

	int old_nfaces = mesh->faces.size();
	int new_nfaces = old_nfaces + nfaces;
	mesh->faces.reserve(new_nfaces);

	// face_len doesn't include the indices themeselves, since that's
	// potentially variable-length
	int face_skip = face_len - face_idx;

	vector<unsigned char> buf(max(face_idx, face_skip));
	vector<int> thisface;
	for (int i = 0; i < nfaces; i++) {
		COND_READ(face_idx > 0, buf[0], face_idx);

		unsigned this_ninds = 3;
		if (face_count >= 0) {
			// Read count - either 1 or 4 bytes
			if (face_idx - face_count == 4) {
				this_ninds = * (unsigned *) &(buf[face_count]);
				if (need_swap)
					swap_unsigned(this_ninds);
			} else {
				this_ninds = buf[face_count];
			}
		}
		thisface.resize(this_ninds);
		COND_READ(true, thisface[0], 4*this_ninds);
		if (need_swap) {
			for (size_t j = 0; j < thisface.size(); j++)
				swap_int(thisface[j]);
		}
		tess(mesh->vertices, thisface, mesh->faces);
		COND_READ(face_skip > 0, buf[0], face_skip);
	}

	return true;
}


// Read a bunch of faces from an ASCII file
static bool read_faces_asc(FILE *f, TriMesh *mesh, int nfaces,
	int face_len, int face_count, int face_idx, bool read_to_eol /* = false */)
{
	if (nfaces < 0 || face_idx < 0)
		return false;

	if (nfaces == 0)
		return true;

	int old_nfaces = mesh->faces.size();
	int new_nfaces = old_nfaces + nfaces;
	mesh->faces.reserve(new_nfaces);

	char buf[1024];
	skip_comments(f);
	TriMesh::dprintf("\n  Reading %d faces... ", nfaces);
	vector<int> thisface;
	for (int i = 0; i < nfaces; i++) {
		thisface.clear();
		int this_face_count = 3;
		for (int j = 0; j < face_len + this_face_count; j++) {
			if (j >= face_idx && j < face_idx + this_face_count) {
				thisface.push_back(0);
				if (!fscanf(f, " %d", &(thisface.back()))) {
					TriMesh::dprintf("Couldn't read vertex index %d for face %d\n",
						j - face_idx, i);
					return false;
				}
			} else if (j == face_count) {
				if (!fscanf(f, " %d", &this_face_count)) {
					TriMesh::dprintf("Couldn't read vertex count for face %d\n", i);
					return false;
				}
			} else {
				fscanf(f, " %s", buf);
			}
		}
		tess(mesh->vertices, thisface, mesh->faces);
		if (read_to_eol) {
			while (1) {
				int c = fgetc(f);
				if (c == EOF || c == '\n')
					break;
			}
		}
	}

	return true;
}


// Read triangle strips from a binary file
static bool read_strips_bin(FILE *f, TriMesh *mesh, bool need_swap)
{
	int striplen;
	COND_READ(true, striplen, 4);
	if (need_swap)
		swap_int(striplen);

	int old_striplen = mesh->tstrips.size();
	int new_striplen = old_striplen + striplen;
	mesh->tstrips.resize(new_striplen);

	TriMesh::dprintf("\n  Reading triangle strips... ");
	COND_READ(true, mesh->tstrips[old_striplen], 4*striplen);
	if (need_swap) {
		for (int i = old_striplen; i < new_striplen; i++)
			swap_int(mesh->tstrips[i]);
	}

	return true;
}


// Read triangle strips from an ASCII file
static bool read_strips_asc(FILE *f, TriMesh *mesh)
{
	skip_comments(f);
	int striplen;
	if (fscanf(f, "%d", &striplen) != 1)
		return false;
	int old_striplen = mesh->tstrips.size();
	int new_striplen = old_striplen + striplen;
	mesh->tstrips.resize(new_striplen);

	TriMesh::dprintf("\n  Reading triangle strips... ");
	skip_comments(f);
	for (int i = old_striplen; i < new_striplen; i++)
		if (fscanf(f, "%d", &mesh->tstrips[i]) != 1)
			return false;

	return true;
}


// Read range grid data from a binary file
static bool read_grid_bin(FILE *f, TriMesh *mesh, bool need_swap)
{
	TriMesh::dprintf("\n  Reading range grid... ");
	int ngrid = mesh->grid_width * mesh->grid_height;
	mesh->grid.resize(ngrid, TriMesh::GRID_INVALID);
	for (int i = 0; i < ngrid; i++) {
		int n = fgetc(f);
		if (n == EOF)
			return false;
		while (n--) {
			if (!fread((void *)&(mesh->grid[i]), 4, 1, f))
				return false;
			if (need_swap)
				swap_int(mesh->grid[i]);
		}
	}

	mesh->triangulate_grid();
	return true;
}


// Read range grid data from an ASCII file
static bool read_grid_asc(FILE *f, TriMesh *mesh)
{
	TriMesh::dprintf("\n  Reading range grid... ");
	int ngrid = mesh->grid_width * mesh->grid_height;
	mesh->grid.resize(ngrid, TriMesh::GRID_INVALID);
	for (int i = 0; i < ngrid; i++) {
		int n;
		if (fscanf(f, "%d", &n) != 1)
			return false;
		while (n--) {
			if (fscanf(f, "%d", &(mesh->grid[i])) != 1)
				return false;
		}
	}

	mesh->triangulate_grid();
	return true;
}


// Parse a PLY property line, and figure how many bytes it represents
// Increments "len" by the number of bytes, or by 1 if !binary
static bool ply_property(const char *buf, int &len, bool binary)
{
	if (LINE_IS("property char") ||
	    LINE_IS("property uchar") ||
	    LINE_IS("property int8") ||
	    LINE_IS("property uint8")) {
		len += 1;
	} else if (LINE_IS("property short") ||
		   LINE_IS("property ushort") ||
		   LINE_IS("property int16") ||
		   LINE_IS("property uint16")) {
		len += (binary ? 2 : 1);
	} else if (LINE_IS("property int") ||
		   LINE_IS("property uint") ||
		   LINE_IS("property float") ||
		   LINE_IS("property int32") ||
		   LINE_IS("property uint32") ||
		   LINE_IS("property float32")) {
		len += (binary ? 4 : 1);
	} else if (LINE_IS("property double") ||
		   LINE_IS("property float64")) {
		len += (binary ? 8 : 1);
	} else {
		fprintf(stderr, "Unsupported vertex property: %s\n", buf);
		return false;
	}
	return true;
}


// Figure out whether this machine is little- or big-endian
static bool we_are_little_endian()
{
	char buf[4];
	*(int *)(&buf[0]) = 1;
	return (buf[0] == 1);
}


// Figure out whether the need_swap setting makes sense, or whether this
// file incorrectly declares its endianness
static void check_need_swap(const point &p, bool &need_swap)
{
	float p0 = p[0], p1 = p[1], p2 = p[2];
	if (need_swap) {
		swap_float(p0);
		swap_float(p1);
		swap_float(p2);
	}
	bool makes_sense = (p0 > -BIGNUM && p0 < BIGNUM &&
			    p1 > -BIGNUM && p1 < BIGNUM &&
			    p2 > -BIGNUM && p2 < BIGNUM);
	if (makes_sense)
		return;

	swap_float(p0);
	swap_float(p1);
	swap_float(p2);

	bool makes_sense_swapped = (p0 > -BIGNUM && p0 < BIGNUM &&
				    p1 > -BIGNUM && p1 < BIGNUM &&
				    p2 > -BIGNUM && p2 < BIGNUM);
	if (makes_sense_swapped) {
		fprintf(stderr, "Compensating for bogus endianness.\n");
		need_swap = !need_swap;
	}
}


// Check whether the indices in the file mistakenly go
// from 1..N instead of 0..N-1
static void check_ind_range(TriMesh *mesh)
{
	if (mesh->faces.empty())
		return;
	int min_ind = mesh->faces[0][0];
	int max_ind = mesh->faces[0][0];
	for (int i = 0; i < mesh->faces.size(); i++) {
		for (int j = 0; j < 3; j++) {
			min_ind = min(min_ind, mesh->faces[i][j]);
			max_ind = max(max_ind, mesh->faces[i][j]);
		}
	}

	int nv = mesh->vertices.size();

	// All good
	if (min_ind == 0 && max_ind == nv-1)
		return;

	// Simple fix: offset everything
	if (max_ind - min_ind == nv-1) {
		TriMesh::dprintf("Found indices ranging from %d through %d\n",
				 min_ind, max_ind);
		TriMesh::dprintf("Remapping to %d through %d\n", 0, nv-1);
		for (int i = 0; i < mesh->faces.size(); i++)
			for (int j = 0; j < 3; j++)
				mesh->faces[i][j] -= min_ind;
		return;
	}

	// Else can't do anything...
}


// Skip comments in an ASCII file (lines beginning with #)
static void skip_comments(FILE *f)
{
	int c;
	bool in_comment = false;
	while (1) {
		c = fgetc(f);
		if (c == EOF)
			return;
		if (in_comment) {
			if (c == '\n')
				in_comment = false;
		} else if (c == '#') {
			in_comment = true;
		} else if (!isspace(c)) {
			break;
		}
	}
	ungetc(c, f);
}


// Tesselate an arbitrary n-gon.  Appends triangles to "tris".
static void tess(const vector<point> &verts, const vector<int> &thisface,
		 vector<TriMesh::Face> &tris)
{
	if (thisface.size() < 3)
		return;
	if (thisface.size() == 3) {
		tris.push_back(TriMesh::Face(thisface[0],
					     thisface[1],
					     thisface[2]));
		return;
	}
	if (thisface.size() == 4) {
		// Triangulate in the direction that
		// gives the shorter diagonal
		const point &p0 = verts[thisface[0]], &p1 = verts[thisface[1]];
		const point &p2 = verts[thisface[2]], &p3 = verts[thisface[3]];
		float d02 = dist2(p0, p2);
		float d13 = dist2(p1, p3);
		int i = (d02 < d13) ? 0 : 1;
		tris.push_back(TriMesh::Face(thisface[i],
					     thisface[(i+1)%4],
					     thisface[(i+2)%4]));
		tris.push_back(TriMesh::Face(thisface[i],
					     thisface[(i+2)%4],
					     thisface[(i+3)%4]));
		return;
	}

	// 5-gon or higher - just tesselate arbitrarily...
	for (int i = 2; i < thisface.size(); i++)
		tris.push_back(TriMesh::Face(thisface[0],
					     thisface[i-1],
					     thisface[i]));
}


// Write mesh to a file
void TriMesh::write(const char *filename)
{
	if (!filename || *filename == '\0')
		return;

	if (vertices.empty()) {
		fprintf(stderr, "Empty mesh - nothing to write!\n");
		return;
	}

	enum { PLY_ASCII, PLY_BINARY_BE, PLY_BINARY_LE,
	       RAY, OBJ, OFF, SM, CC } filetype;
	// Set default file type to be native-endian binary ply
	filetype = we_are_little_endian() ? PLY_BINARY_LE : PLY_BINARY_BE;
	bool write_norm = false;
	bool float_color = false;

	// Infer file type from file extension
	const char *c = strrchr(filename, '.');
	if (c) {
		if (!strncasecmp(c, ".ply", 4))
			filetype = we_are_little_endian() ?
					PLY_BINARY_LE :
					PLY_BINARY_BE;
		else if (!strncasecmp(c, ".ray", 4))
			filetype = RAY;
		else if (!strncasecmp(c, ".obj", 4))
			filetype = OBJ;
		else if (!strncasecmp(c, ".off", 4))
			filetype = OFF;
		else if (!strncasecmp(c, ".sm", 3))
			filetype = SM;
		else if (!strncasecmp(c, ".cc", 3))
			filetype = CC;
		else if (!strncasecmp(c, ".c++", 4))
			filetype = CC;
		else if (!strncasecmp(c, ".cpp", 4))
			filetype = CC;
		else if (!strncasecmp(c, ".C", 2))
			filetype = CC;
	}

	// Handle filetype:filename.foo constructs
	while (1) {
		if (!strncasecmp(filename, "norm:", 5)) {
			filename += 5;
			write_norm = true;
		} else if (!strncasecmp(filename, "cflt:", 5)) {
			filename += 5;
			float_color = true;
		} else if (!strncasecmp(filename, "ply:", 4)) {
			filename += 4;
			filetype = we_are_little_endian() ?
					PLY_BINARY_LE :
					PLY_BINARY_BE;
		} else if (!strncasecmp(filename, "ply_binary:", 11)) {
			filename += 11;
			filetype = we_are_little_endian() ?
					PLY_BINARY_LE :
					PLY_BINARY_BE;
		} else if (!strncasecmp(filename, "ply_binary_be:", 14)) {
			filename += 14;
			filetype = PLY_BINARY_BE;
		} else if (!strncasecmp(filename, "ply_binary_le:", 14)) {
			filename += 14;
			filetype = PLY_BINARY_LE;
		} else if (!strncasecmp(filename, "ply_ascii:", 10)) {
			filename += 10;
			filetype = PLY_ASCII;
		} else if (!strncasecmp(filename, "ply_asc:", 8)) {
			filename += 8;
			filetype = PLY_ASCII;
		} else if (!strncasecmp(filename, "ascii:", 6)) {
			filename += 6;
			filetype = PLY_ASCII;
		} else if (!strncasecmp(filename, "asc:", 4)) {
			filename += 4;
			filetype = PLY_ASCII;
		} else if (!strncasecmp(filename, "be:", 3)) {
			filename += 3;
			filetype = PLY_BINARY_BE;
		} else if (!strncasecmp(filename, "le:", 3)) {
			filename += 3;
			filetype = PLY_BINARY_LE;
		} else if (!strncasecmp(filename, "ray:", 4)) {
			filename += 4;
			filetype = RAY;
		} else if (!strncasecmp(filename, "obj:", 4)) {
			filename += 4;
			filetype = OBJ;
		} else if (!strncasecmp(filename, "off:", 4)) {
			filename += 4;
			filetype = OFF;
		} else if (!strncasecmp(filename, "sm:", 3)) {
			filename += 3;
			filetype = SM;
		} else {
			break;
		}
	}


	FILE *f = NULL;

	if (strcmp(filename, "-") == 0) {
		f = stdout;
		filename = "standard output";
	} else {
		f = fopen(filename, "wb");
		if (!f) {
			perror("fopen");
			fprintf(stderr, "Error opening %s for writing.\n", filename);
			return;
		}
	}

	dprintf("Writing %s... ", filename);

	switch (filetype) {
		case PLY_ASCII:
			write_ply_ascii(this, f, write_norm, float_color);
			break;
		case PLY_BINARY_BE:
			write_ply_binary(this, f,
				we_are_little_endian(), write_norm, float_color);
			break;
		case PLY_BINARY_LE:
			write_ply_binary(this, f,
				!we_are_little_endian(), write_norm, float_color);
			break;
		case RAY:
			write_ray(this, f);
			break;
		case OBJ:
			write_obj(this, f);
			break;
		case OFF:
			write_off(this, f);
			break;
		case SM:
			write_sm(this, f);
			break;
		case CC:
			write_cc(this, f, filename, write_norm, float_color);
			break;
	}
	fclose(f);
	dprintf("Done.\n");
}


// Write a ply header
static void write_ply_header(TriMesh *mesh, FILE *f, const char *format,
			     bool write_grid, bool write_tstrips,
			     bool write_norm, bool float_color)
{
	fprintf(f, "ply\nformat %s 1.0\n", format);
	if (write_grid) {
		fprintf(f, "obj_info num_cols %d\n", mesh->grid_width);
		fprintf(f, "obj_info num_rows %d\n", mesh->grid_height);
	}
	fprintf(f, "element vertex %lu\n",
		(unsigned long) mesh->vertices.size());
	fprintf(f, "property float x\n");
	fprintf(f, "property float y\n");
	fprintf(f, "property float z\n");
	if (write_norm && !mesh->normals.empty()) {
		fprintf(f, "property float nx\n");
		fprintf(f, "property float ny\n");
		fprintf(f, "property float nz\n");
	}
	if (!mesh->colors.empty() && float_color) {
		fprintf(f, "property float diffuse_red\n");
		fprintf(f, "property float diffuse_green\n");
		fprintf(f, "property float diffuse_blue\n");
	}
	if (!mesh->colors.empty() && !float_color) {
		fprintf(f, "property uchar diffuse_red\n");
		fprintf(f, "property uchar diffuse_green\n");
		fprintf(f, "property uchar diffuse_blue\n");
	}
	if (!mesh->confidences.empty()) {
		fprintf(f, "property float confidence\n");
	}
	if (write_grid) {
		int ngrid = mesh->grid_width * mesh->grid_height;
		fprintf(f, "element range_grid %d\n", ngrid);
		fprintf(f, "property list uchar int vertex_indices\n");
	} else if (write_tstrips) {
		fprintf(f, "element tristrips 1\n");
		fprintf(f, "property list int int vertex_indices\n");
	} else {
		fprintf(f, "element face %lu\n",
			(unsigned long) mesh->faces.size());
		fprintf(f, "property list uchar int vertex_indices\n");
	}
	fprintf(f, "end_header\n");
}


// Write an ASCII ply file
static void write_ply_ascii(TriMesh *mesh, FILE *f, bool write_norm,
	bool float_color)
{
	if (write_norm)
		mesh->need_normals();

	bool write_grid = !mesh->grid.empty();
	bool write_tstrips = !write_grid && !mesh->tstrips.empty();

	write_ply_header(mesh, f, "ascii", write_grid, write_tstrips,
		write_norm, float_color);
	write_verts_asc(mesh, f, "", write_norm ? " " : 0, " ", float_color,
		" ", "");
	if (write_grid) {
		write_grid_asc(mesh, f);
	} else if (write_tstrips) {
		fprintf(f, "%lu ", (unsigned long) mesh->tstrips.size());
		mesh->convert_strips(TriMesh::TSTRIP_TERM);
		write_strips_asc(mesh, f);
		mesh->convert_strips(TriMesh::TSTRIP_LENGTH);
	} else {
		write_faces_asc(mesh, f, "3 ", "");
	}
}


// Write a binary ply file
static void write_ply_binary(TriMesh *mesh, FILE *f,
	bool need_swap, bool write_norm, bool float_color)
{
	if (write_norm)
		mesh->need_normals();
	const char *format = (need_swap ^ we_are_little_endian()) ?
		"binary_little_endian" : "binary_big_endian";

	bool write_grid = !mesh->grid.empty();
	bool write_tstrips = !write_grid && !mesh->tstrips.empty();

	write_ply_header(mesh, f, format, write_grid, write_tstrips,
		write_norm, float_color);
	write_verts_bin(mesh, f, need_swap, write_norm, true, float_color, true);
	if (write_grid) {
		write_grid_bin(mesh, f, need_swap);
	} else if (write_tstrips) {
		int s = mesh->tstrips.size();
		if (need_swap)
			swap_int(s);
		fwrite(&s, 4, 1, f);
		mesh->convert_strips(TriMesh::TSTRIP_TERM);
		write_strips_bin(mesh, f, need_swap);
		mesh->convert_strips(TriMesh::TSTRIP_LENGTH);
	} else {
		char buf[1] = { 3 };
		write_faces_bin(mesh, f, need_swap, 1, buf, 0, 0);
	}
}


// Write a ray file
static void write_ray(TriMesh *mesh, FILE *f)
{
	fprintf(f, "#camera 0 0 1 0 0 -1 0 1 0 0.2\n");
	fprintf(f, "#background 0 0 0\n");
	fprintf(f, "#ambient 0 0 0\n");
	fprintf(f, "#material_num 1\n");
	fprintf(f, "#material 0 0 0  1 1 1  0 0 0  0 0 0  0 0 1  -1  !!\n");
	fprintf(f, "#vertex_num %lu\n", (unsigned long) mesh->vertices.size());
	mesh->need_normals();
	write_verts_asc(mesh, f, "#vertex ", "  ", 0, false, 0, "  0 0");
	mesh->need_faces();
	write_faces_asc(mesh, f, "#shape_triangle 0  ", "");
}


// Write a obj file
static void write_obj(TriMesh *mesh, FILE *f)
{
	fprintf(f, "# OBJ\n");
	write_verts_asc(mesh, f, "v ", 0, 0, false, 0, "");
	mesh->need_faces();
	for (int i = 0; i < mesh->faces.size(); i++) {
		mesh->faces[i][0]++;
		mesh->faces[i][1]++;
		mesh->faces[i][2]++;
	}
	write_faces_asc(mesh, f, "f ", "");
	for (int i = 0; i < mesh->faces.size(); i++) {
		mesh->faces[i][0]--;
		mesh->faces[i][1]--;
		mesh->faces[i][2]--;
	}
}


// Write a off file
static void write_off(TriMesh *mesh, FILE *f)
{
	fprintf(f, "OFF\n");
	mesh->need_faces();
	fprintf(f, "%lu %lu 0\n", (unsigned long) mesh->vertices.size(),
		(unsigned long) mesh->faces.size());
	write_verts_asc(mesh, f, "", 0, 0, false, 0, "");
	write_faces_asc(mesh, f, "3 ", "");
}


// Write an SM file
static void write_sm(TriMesh *mesh, FILE *f)
{
	fprintf(f, "%lu\n", (unsigned long) mesh->vertices.size());
	write_verts_asc(mesh, f, "", 0, 0, false, 0, "");
	mesh->need_faces();
	fprintf(f, "%lu\n", (unsigned long) mesh->faces.size());
	write_faces_asc(mesh, f, "", "");
	fprintf(f, "0 0\n");
}


// Convert colors float -> uchar
static unsigned char color2uchar(float p)
{
	return min(max(int(255.0f * p + 0.5f), 0), 255);
}


// Write C++ code
static void write_cc(TriMesh *mesh, FILE *f, const char *filename,
	bool write_norm, bool float_color)
{
	mesh->need_faces();
	if (write_norm)
		mesh->need_normals();

	char *meshname = new char[strlen(filename)+1];
	strcpy(meshname, filename);
	char *c = strrchr(meshname, '.');
	if (c)
		*c = '\0';
	fprintf(f, "#include <string.h>\n");
	fprintf(f, "#include \"TriMesh.h\"\n\n");
	fprintf(f, "TriMesh *make_%s()\n{", meshname);
	delete [] meshname;

	fprintf(f, "\tstatic const float vertdata[][3] = {\n");
	int nv = mesh->vertices.size(), nf = mesh->faces.size();
	for (int i = 0; i < nv; i++) {
		fprintf(f, "\t\t{ %.7g, %.7g, %.7g },\n",
				mesh->vertices[i][0],
				mesh->vertices[i][1],
				mesh->vertices[i][2]);
	}
	fprintf(f, "\t};\n");
	if (write_norm) {
		fprintf(f, "\tstatic const float normdata[][3] = {\n");
		for (int i = 0; i < nv; i++) {
			fprintf(f, "\t\t{ %.7g, %.7g, %.7g },\n",
					mesh->normals[i][0],
					mesh->normals[i][1],
					mesh->normals[i][2]);
		}
		fprintf(f, "\t};\n");
	}
	if (!mesh->colors.empty() && float_color) {
		fprintf(f, "\tstatic const float colordata[][3] = {\n");
		for (int i = 0; i < nv; i++) {
			fprintf(f, "\t\t{ %.7g, %.7g, %.7g },\n",
					mesh->colors[i][0],
					mesh->colors[i][1],
					mesh->colors[i][2]);
		}
		fprintf(f, "\t};\n");
	}
	if (!mesh->colors.empty() && !float_color) {
		fprintf(f, "\tstatic const unsigned char colordata[][3] = {\n");
		for (int i = 0; i < nv; i++) {
			fprintf(f, "\t\t{ %d, %d, %d },\n",
					color2uchar(mesh->colors[i][0]),
					color2uchar(mesh->colors[i][1]),
					color2uchar(mesh->colors[i][2]));
		}
		fprintf(f, "\t};\n");
	}
	fprintf(f, "\tstatic const int facedata[][3] = {\n");
	for (int i = 0; i < nf; i++) {
		fprintf(f, "\t\t{ %d, %d, %d },\n",
				mesh->faces[i][0],
				mesh->faces[i][1],
				mesh->faces[i][2]);
	}
	fprintf(f, "\t};\n");
	fprintf(f, "\n\tTriMesh *m = new TriMesh;\n");
	fprintf(f, "\tm->vertices.resize(%d);\n", nv);
	fprintf(f, "\tmemcpy(&m->vertices[0][0], vertdata, sizeof(vertdata));\n");
	if (!mesh->colors.empty()) {
		fprintf(f, "\tm->colors.resize(%d);\n", nv);
		fprintf(f, "\tmemcpy(&m->colors[0][0], colordata, sizeof(colordata));\n");
	}
	if (write_norm) {
		fprintf(f, "\tm->normals.resize(%d);\n", nv);
		fprintf(f, "\tmemcpy(&m->normals[0][0], normdata, sizeof(normdata));\n");
	}
	fprintf(f, "\tm->faces.resize(%d);\n", nf);
	fprintf(f, "\tmemcpy(&m->faces[0][0], facedata, sizeof(facedata));\n");
	fprintf(f, "\n\treturn m;\n");
	fprintf(f, "}\n");
}


// Write a bunch of vertices to an ASCII file
static void write_verts_asc(TriMesh *mesh, FILE *f,
			    const char *before_vert,
			    const char *before_norm,
			    const char *before_color,
			    bool float_color,
			    const char *before_conf,
			    const char *after_line)
{
    for (int i = 0; i < mesh->vertices.size(); i++) {
		fprintf(f, "%s%.7g %.7g %.7g", before_vert,
				mesh->vertices[i][0],
				mesh->vertices[i][1],
				mesh->vertices[i][2]);
		if (!mesh->normals.empty() && before_norm)
			fprintf(f, "%s%.7g %.7g %.7g", before_norm,
				mesh->normals[i][0],
				mesh->normals[i][1],
				mesh->normals[i][2]);
		if (!mesh->colors.empty() && before_color && float_color)
			fprintf(f, "%s%.7g %.7g %.7g", before_color,
				mesh->colors[i][0],
				mesh->colors[i][1],
				mesh->colors[i][2]);
		if (!mesh->colors.empty() && before_color && !float_color)
			fprintf(f, "%s%d %d %d", before_color,
				color2uchar(mesh->colors[i][0]),
				color2uchar(mesh->colors[i][1]),
				color2uchar(mesh->colors[i][2]));
		if (!mesh->confidences.empty() && before_conf)
			fprintf(f, "%s%.7g", before_conf, mesh->confidences[i]);
		fprintf(f, "%s\n", after_line);
	}
}


// Write a bunch of vertices to a binary file
static void write_verts_bin(TriMesh *mesh, FILE *f, bool need_swap,
			    bool write_norm, bool write_color,
			    bool float_color, bool write_conf)
{
	if (need_swap) {
		for (int i = 0; i < mesh->vertices.size(); i++) {
			swap_float(mesh->vertices[i][0]);
			swap_float(mesh->vertices[i][1]);
			swap_float(mesh->vertices[i][2]);
		}
		if (!mesh->normals.empty()) {
			for (int i = 0; i < mesh->normals.size(); i++) {
				swap_float(mesh->normals[i][0]);
				swap_float(mesh->normals[i][1]);
				swap_float(mesh->normals[i][2]);
			}
		}
		if (!mesh->colors.empty() && float_color) {
			for (int i = 0; i < mesh->normals.size(); i++) {
				swap_float(mesh->colors[i][0]);
				swap_float(mesh->colors[i][1]);
				swap_float(mesh->colors[i][2]);
			}
		}
		if (!mesh->confidences.empty()) {
			for (int i = 0; i < mesh->confidences.size(); i++)
				swap_float(mesh->confidences[i]);
		}
	}
	if ((mesh->normals.empty() || !write_norm) &&
	    (mesh->colors.empty() || !write_color) &&
	    (mesh->confidences.empty() || !write_conf)) {
		// Optimized vertex-only code
		fwrite(&(mesh->vertices[0][0]), 12*mesh->vertices.size(), 1, f);
	} else {
		// Generic code
		for (int i = 0; i < mesh->vertices.size(); i++) {
			fwrite(&(mesh->vertices[i][0]), 12, 1, f);
			if (!mesh->normals.empty() && write_norm)
				fwrite(&(mesh->normals[i][0]), 12, 1, f);
			if (!mesh->colors.empty() && write_color && float_color)
				fwrite(&(mesh->colors[i][0]), 12, 1, f);
			if (!mesh->colors.empty() && write_color && !float_color) {
				char c[3] = {
					color2uchar(mesh->colors[i][0]),
					color2uchar(mesh->colors[i][1]),
					color2uchar(mesh->colors[i][2]) };
				fwrite(&c, 3, 1, f);
			}
			if (!mesh->confidences.empty() && write_conf)
				fwrite(&(mesh->confidences[i]), 4, 1, f);
		}
	}
	if (need_swap) {
		for (int i = 0; i < mesh->vertices.size(); i++) {
			swap_float(mesh->vertices[i][0]);
			swap_float(mesh->vertices[i][1]);
			swap_float(mesh->vertices[i][2]);
		}
		if (!mesh->normals.empty()) {
			for (int i = 0; i < mesh->normals.size(); i++) {
				swap_float(mesh->normals[i][0]);
				swap_float(mesh->normals[i][1]);
				swap_float(mesh->normals[i][2]);
			}
		}
		if (!mesh->colors.empty() && float_color) {
			for (int i = 0; i < mesh->normals.size(); i++) {
				swap_float(mesh->colors[i][0]);
				swap_float(mesh->colors[i][1]);
				swap_float(mesh->colors[i][2]);
			}
		}
		if (!mesh->confidences.empty()) {
			for (int i = 0; i < mesh->confidences.size(); i++)
				swap_float(mesh->confidences[i]);
		}
	}
}


// Write a bunch of faces to an ASCII file
static void write_faces_asc(TriMesh *mesh, FILE *f,
			    const char *before_face, const char *after_line)
{
	mesh->need_faces();
	for (int i = 0; i < mesh->faces.size(); i++) {
		fprintf(f, "%s%d %d %d%s\n", before_face, mesh->faces[i][0],
			mesh->faces[i][1], mesh->faces[i][2], after_line);
	}
}


// Write a bunch of faces to a binary file
static void write_faces_bin(TriMesh *mesh, FILE *f, bool need_swap,
			    int before_face_len, const char *before_face,
			    int after_face_len, const char *after_face)
{
	mesh->need_faces();
	if (need_swap) {
		for (int i = 0; i < mesh->faces.size(); i++) {
			swap_int(mesh->faces[i][0]);
			swap_int(mesh->faces[i][1]);
			swap_int(mesh->faces[i][2]);
		}
	}
	for (int i = 0; i < mesh->faces.size(); i++) {
		if (before_face_len)
			fwrite(before_face, before_face_len, 1, f);
		fwrite(&(mesh->faces[i][0]), 12, 1, f);
		if (after_face_len)
			fwrite(after_face, after_face_len, 1, f);
	}
	if (need_swap) {
		for (int i = 0; i < mesh->faces.size(); i++) {
			swap_int(mesh->faces[i][0]);
			swap_int(mesh->faces[i][1]);
			swap_int(mesh->faces[i][2]);
		}
	}
}


// Write tstrips to an ASCII file
static void write_strips_asc(TriMesh *mesh, FILE *f)
{
	for (int i = 0; i < mesh->tstrips.size(); i++) {
		fprintf(f, "%d ", mesh->tstrips[i]);
	}
	fprintf(f, "\n");
}


// Write tstrips to a binary file
static void write_strips_bin(TriMesh *mesh, FILE *f, bool need_swap)
{
	if (need_swap) {
		for (int i = 0; i < mesh->tstrips.size(); i++)
			swap_int(mesh->tstrips[i]);
	}
	fwrite(&(mesh->tstrips[0]), 4*mesh->tstrips.size(), 1, f);
	if (need_swap) {
		for (int i = 0; i < mesh->tstrips.size(); i++)
			swap_int(mesh->tstrips[i]);
	}
}


// Write range grid to an ASCII file
static void write_grid_asc(TriMesh *mesh, FILE *f)
{
	for (int i = 0; i < mesh->grid.size(); i++) {
		if (mesh->grid[i] < 0)
			fprintf(f, "0\n");
		else
			fprintf(f, "1 %d\n", mesh->grid[i]);
	}
}


// Write range grid to a binary file
static void write_grid_bin(TriMesh *mesh, FILE *f, bool need_swap)
{
	unsigned char zero = 0;
	unsigned char one = 1;
	for (int i = 0; i < mesh->grid.size(); i++) {
		if (mesh->grid[i] < 0) {
			fwrite(&zero, 1, 1, f);
		} else {
			fwrite(&one, 1, 1, f);
			int g = mesh->grid[i];
			if (need_swap)
				swap_int(g);
			fwrite(&g, 4, 1, f);
		}
	}
}


// Debugging printout, controllable by a "verbose"ness parameter
int TriMesh::verbose = 1;

void TriMesh::set_verbose(int verbose_)
{
	verbose = verbose_;
}

int TriMesh::dprintf(const char *format, ...)
{
	if (!verbose)
		return 0;

	va_list ap;
	va_start(ap, format);
	int ret = vfprintf(stderr, format, ap);
	va_end(ap);
	fflush(stderr);
	return ret;
}

