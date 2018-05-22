#pragma once

#ifdef WIN32
	#define WIN32_LEAN_AND_MEAN
	#include <windows.h>
#endif

#include <stdlib.h>
#include <vector>

#include "Primitive3D.h"
#include "Vector2D.h"
#include "Vector3D.h"
#include "BBox3D.h"
#include "BSphere3D.h"
#include "Triangle3D.h"
#include "SpatialSubdivision.h"
#include "Material3D.h"

using namespace std;

class PrimitiveGroup3D
{
public:
#ifdef BUFFER_OBJECT
	Face3D *faces;
	Vector3D *fc_normals;
	unsigned long numfc_normals;
	unsigned long memfc_normals;
#else // COMPILE_LIST
	Triangle3D *faces;
#endif
	unsigned long numfaces;
	unsigned long memfaces;     // memory allocated
	unsigned int  mtrlIdx;
	bool has_normals;
	bool has_texcoords[3];
	char name[MAXSTRING];
	GLuint pibo;                // polygon buffer object
	
	PrimitiveGroup3D(void)
	{
		faces = NULL;
		numfaces = 0;
#ifdef BUFFER_OBJECT
		fc_normals = NULL;
		numfc_normals = 0;
		memfc_normals = 0;
#endif
		memfaces = 0;
		has_normals = false;
		has_texcoords[0] = has_texcoords[1] = has_texcoords[2] = false;
	}
	
	~PrimitiveGroup3D(void)
	{
		if (faces != NULL)
		    free (faces);
		numfaces = 0;
		memfaces = 0;
#ifdef BUFFER_OBJECT
		if (fc_normals != NULL)
		    free (fc_normals);
		numfc_normals = 0;
		memfc_normals = 0;
#endif
	}

	void dumpAll (void)
	{
		unsigned long i;

        printf ("\tGroup \"%s\" info:\n", name);
        printf ("\t\tnum of Faces : %lu\n", numfaces);
        printf ("\t\tmaterial_id : %2d\n", mtrlIdx);
        printf ("\t\thas_texcoords[0] : %s \thas_texcoords[1] : %s \thas_texcoords[2] : %s\n", has_texcoords[0] ? "yes" : "no", has_texcoords[1] ? "yes" : "no", has_texcoords[2] ? "yes" : "no");

        for (i = 0; i < numfaces; i++)
        {
#ifdef BUFFER_OBJECT
            Face3D *t = &(faces[i]);

            printf ("\t\tv : %4ld %4ld %4ld\n",
                t->vertIdx[0], t->vertIdx[1], t->vertIdx[2]);
#else // COMPILE_LIST
            Triangle3D *t = &(faces[i]);

            printf ("\t\tv : %4ld %4ld %4ld\t"
                    "\t\tn : %4ld %4ld %4ld\t"
                    "\t\tt : %4ld %4ld %4ld\t"
                    "\t\tnormal : % f % f % f\n",
                t->vertIdx[0], t->vertIdx[1], t->vertIdx[2],
                t->normIdx[0], t->normIdx[1], t->normIdx[2],
                t->texcIdx[0], t->texcIdx[1], t->texcIdx[2],
                t->fc_normal[0], t->fc_normal[1], t->fc_normal[2]);
#endif
        }
	}

	void dump (void)
	{
        printf ("\tGroup \"%s\" info:\n", name);
        printf ("\t\tnum of Faces : %lu\n", numfaces);
        printf ("\t\tmaterial_id : %2d\n", mtrlIdx);
        printf ("\t\thas_texcoords[0] : %s \thas_texcoords[1] : %s \thas_texcoords[2] : %s\n", has_texcoords[0] ? "yes" : "no", has_texcoords[1] ? "yes" : "no", has_texcoords[2] ? "yes" : "no");
	}
};

class Mesh3D : public Primitive3D
{
public:
	Mesh3D(void);
	Mesh3D(char *filename);
	Mesh3D(char *filename, vector<char *>searchpaths);
	virtual ~Mesh3D(void);
	
	virtual bool intersectAll(Ray3D &r);
	virtual bool intersect(Ray3D &r);
	
	float getMaxDimension(void);
	Vector3D getCenter(void) { return center; }
	virtual void draw(void);
	virtual void drawOpaque(void);
	virtual void drawTransparent(void);
	void addSearchPaths(vector<char *>addpaths);
	void addSearchPath(char * addpath);
	BBox3D getBoundingBox(void) {return bbox;}
	BSphere3D getBoundingSphere(void) {return bsphere;}
	BSphere3D getTightBoundingSphere(void) {return btsphere;}
	void setDoubleSided(bool ds) {double_sided = ds;}
	        void dump(void);
	        void dumpAll(void);
	unsigned int findMaterial(char *name);
	Material3D * getMaterials(void) { return materials; }
	unsigned long getNumMaterials(void) { return nummaterials; }
	
	void readFormat(const char *filename);
	void writeFormat(const char *filename);
	void processData(void);
	char *getName(void) { return filename; }
	
	// geometry structuring methods
	void calcBBox(void);
	void calcBSphere(void); // computes bounding sphere of bounding box
	void calcTightBSphere(void);
	void prepareFaceNormals(void);
	void prepareVertexTangents(void);
#ifdef BUFFER_OBJECT
	void bufferObject(void);
#else // COMPILE_LIST
	void compileLists(void);
#endif
	
protected:
	// internal calculation methods
	void calcFaceNormal(Vector3D *nrm,Vector3D &v1,Vector3D &v2,Vector3D &v3,bool normalized);
	
	// from obj loader
	void drawProp(int mi, int ri);
	void drawVert(void);
	void drawMtrl(int mi);
	void drawSurf(unsigned long si);
	void drawObject(void);
	
	// internal query methods
	char *getFullPath(const char * file);
	
	char filename[MAXSTRING];
	char matlib[MAXSTRING];
	
	// maps
	unsigned int loadTexture(const char *filename);
	
	vector <char *> paths;
	BBox3D bbox;
	BSphere3D bsphere;
	BSphere3D btsphere;
	SpatialSubdivision *spatialSubd;
	Vector3D center;
	bool double_sided;
	
	unsigned int dlist_opaque;
	unsigned int dlist_transparent;
	
public:
	Vector3D *vertices;
	Vector3D *normals;
	Vector3D *tangents;
	Vector2D *texcoords[3];
	Material3D *materials;
	PrimitiveGroup3D *groups;
	
	long unsigned int numfaces;
	long unsigned int numvertices;
	long unsigned int numnormals;
	long unsigned int numtangents;
	long unsigned int numtexcoords[3];
	long unsigned int numgroups;
	long unsigned int nummaterials;
	
#ifdef BUFFER_OBJECT
    GLuint bobject_opaque;      // vertex buffer object

    GLintptr ptrtangents;
    GLintptr ptrnormals;
    GLintptr ptrtexcoords;
    GLintptr ptrvertices;

    // memory allocated
    unsigned long memtangents;
    unsigned long memnormals;
    unsigned long memtexcoords[3];
    unsigned long memvertices;

    unsigned long memmaterials;
    unsigned long memgroups;
#endif
};

