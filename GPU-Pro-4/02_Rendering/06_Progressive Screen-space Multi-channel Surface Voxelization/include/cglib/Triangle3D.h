#pragma once

#include "Primitive3D.h"

class Triangle3D : public Primitive3D
{
public:
	Triangle3D(void);
	virtual ~Triangle3D(void);

	virtual bool intersect(Ray3D &r, Vector3D * vertexlist, Vector3D * normallist);
	virtual bool intersect(Ray3D &r, Vector3D * vertexlist);
	long unsigned int vertIdx[3];
	long unsigned int normIdx[3];
	long unsigned int texcIdx[3]; 
	Vector3D fc_normal;
};

typedef struct
{
	long unsigned int vertIdx[3];
}
Face3D;

bool FindIntersection(Triangle3D triangle, Ray3D &r, Vector3D * vertexlist, Vector3D * normallist);

