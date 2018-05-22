#pragma once

#include "Vector3D.h"
#include "Ray3D.h"
#include "Primitive3D.h"
#include "Triangle3D.h"

class BoxBin
{
public:
	Vector3D min, max;
	Triangle3D * triangleList;
	long triangles;
	Vector3D * vertexlist;
	Vector3D * normallist;
	virtual bool intersect(Ray3D &r);
	bool intersectTriangle(Triangle3D tr);
	bool intersectRayBox (Vector3D Orig, Vector3D End, Vector3D * Qin, Vector3D * Qout, int IsSegment);
	void fillBin(Triangle3D *tr, long numTris);
	BoxBin(Vector3D bmin, Vector3D bmax, Vector3D * vertexlist, Vector3D * normallist);
	virtual ~BoxBin();
};

class SpatialSubdivision
{
public:
	int dimx, dimy, dimz;
    class BoxBin ** bins;
	Vector3D min, max;
    virtual bool intersect(Ray3D &r);
	SpatialSubdivision(Triangle3D *tr, long numTris, Vector3D * vertexlist, long numVertices, 
						Vector3D * normallist, long numNormals, int dimx, int dimy, int dimz );
	SpatialSubdivision();
	virtual ~SpatialSubdivision();
};

