#pragma once

#include "Ray3D.h"

class Primitive3D
{
public:
	Primitive3D(void);
	virtual ~Primitive3D(void);
	virtual bool intersect(Ray3D r);
};
