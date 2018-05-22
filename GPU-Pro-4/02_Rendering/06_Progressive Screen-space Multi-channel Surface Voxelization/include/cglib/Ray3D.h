#pragma once

#include "Vector3D.h"
#include "Matrix4D.h"

class RaySignature3D
{
private:
	static unsigned long _ray_counter_h;
	static unsigned long _ray_counter_l;
	unsigned long signature_l, signature_h;
	unsigned int threadID;
public:
	RaySignature3D(unsigned int thread_id=0);
	bool operator== (RaySignature3D);
	unsigned long getSignatureH() {return signature_h;}
	unsigned long getSignatureL() {return signature_l;}
};

class IsectPoint3D
{
public:
	IsectPoint3D();
	DBL t;
	bool inside;
	Vector3D p_isect;
	Vector3D n_isect;
	void * pData;
	class Primitive3D *primitive;
};

class Ray3D
{
protected:
	RaySignature3D sig;
public:
	Vector3D origin;
	Vector3D dir;
	DBL t;
	DBL strength;
	bool inside;
	bool hit;
	int depth;
	Vector3D p_isect;
	Vector3D n_isect;
	void * pData;
	class Primitive3D *primitive;
	void xform(Matrix4D m);
	Ray3D operator= (Ray3D r);
	Ray3D(void);
	Ray3D(Vector3D start, Vector3D direction, unsigned int thread_id = 0);
	RaySignature3D getSignature() {return sig;}
	~Ray3D(void);
};

