
#ifdef WIN32
	#define WIN32_LEAN_AND_MEAN
	#include <windows.h>
#endif
#include <stdlib.h>
#include <string.h>
#include <float.h>

#include "SpatialSubdivision.h"

SpatialSubdivision::SpatialSubdivision()
{
	dimx = dimy = dimz = 0;
	bins = NULL;
	min.x = min.y = min.z = max.x = max.y = max.z = 0;
}

SpatialSubdivision::SpatialSubdivision(Triangle3D *tr, long numTris, Vector3D * vertexlist, long numVertices, Vector3D * normallist, long numNormals, int dimxnew, int dimynew, int dimznew )
{
	dimx = dimxnew;
	dimy = dimynew;
	dimz = dimznew;

	min.x =100000;
	min.y =100000;
	min.z =100000;
	max.x =-100000;
	max.y =-100000;
	max.z =-100000;
	long i;
	for(i = 1; i<numVertices; i++)
	{
		if(vertexlist[i].x>max.x)
			max.x = vertexlist[i].x;
		if(vertexlist[i].y>max.y)
			max.y = vertexlist[i].y;
		if(vertexlist[i].z>max.z)
			max.z = vertexlist[i].z;

		if(vertexlist[i].x<min.x)
			min.x = vertexlist[i].x;
		if(vertexlist[i].y<min.y)
			min.y = vertexlist[i].y;
		if(vertexlist[i].z<min.z)
			min.z = vertexlist[i].z;
		
	}
	long dimxyz = dimx*dimy*dimz;
	int j, k;
	Vector3D incr;
	incr.x= (max.x - min.x)/dimx;
	incr.y = (max.y - min.y)/dimy;
	incr.z = (max.z - min.z)/dimz;
	Vector3D cur, next;

	bins = (BoxBin **)malloc(dimxyz*sizeof(BoxBin *));
	
	for(k=0; k< dimz; k++)
	for(j=0; j< dimy; j++)
	for(i=0; i< dimx; i++)
	{
		cur.x=i*incr.x + min.x;
		cur.y=j*incr.y + min.y;
		cur.z=k*incr.z + min.z;
        next.x = min.x + (i+1)*incr.x;
	    next.y = min.y + (j+1)*incr.y;
	    next.z = min.z + (k+1)*incr.z;
	
		bins[i+j*dimx+k*dimx*dimy]=new BoxBin(cur, next, vertexlist, normallist );
		bins[i+j*dimx+k*dimx*dimy]->fillBin(tr, numTris);
	}
}

SpatialSubdivision::~SpatialSubdivision()
{
	for(long i=0; i< dimx*dimy*dimz; i++)
	{
		delete bins[i];
	}
	free (bins);
}

bool SpatialSubdivision::intersect(Ray3D &r)
{
	int i;
	float nearestT = 100000;
	Vector3D iPoint;
	
	for(i=0; i<dimx*dimy*dimz; i++)
	{
		if(bins[i]->intersect(r))
			if(nearestT>r.t)
			{
				nearestT = r.t;
				iPoint = r.p_isect;

			}
	}

	if (nearestT<100000)
	{
		r.t = nearestT;
		r.p_isect = iPoint;
		return true;
	}
	return false;
}

bool BoxBin::intersect(Ray3D &r)
{
	long i;
	Vector3D Qin, Qout;
	float nearestT = 100000;
	Vector3D iPoint;
	if(! intersectRayBox (r.origin, r.origin + r.dir, &Qin, &Qout, 0))
		return false;
	for(i=0; i<triangles; i++)
	{
		if(triangleList[i].intersect(r, vertexlist, normallist))
			if(nearestT>r.t)
			{
				nearestT = r.t;
				iPoint = r.p_isect;

			}
	}
	if (nearestT<100000)
	{
		r.t = nearestT;
		r.p_isect = iPoint;
		return true;
	}
	return false;
}

BoxBin::BoxBin(Vector3D bmin, Vector3D bmax, Vector3D * vertlist, Vector3D * normlist)
{
	min = bmin;
	max = bmax;
	vertexlist = vertlist;
	normallist = normlist;
	triangles = 0;
}

BoxBin::~BoxBin()
{
	free(triangleList);
}

void BoxBin::fillBin(Triangle3D *tr, long numTris)
{
	Triangle3D* tmpTriList = (Triangle3D *)malloc(numTris*sizeof(Triangle3D));

	long triCount = 0;
	long m;
	
	for(m=0; m<numTris; m++)
	{
		if(intersectTriangle(tr[m]))
		{
			tmpTriList[triCount]=tr[m];
			triCount++;
		}	
	}
	if (triCount==0)
	{
		free (tmpTriList);
		triangles = 0;
		return;
	}
	triangleList = (Triangle3D *)malloc(triCount*sizeof(Triangle3D));
	memcpy(triangleList,tmpTriList,triCount*sizeof(Triangle3D));
	triangles = triCount;
	free(tmpTriList);
}

bool BoxBin::intersectTriangle(Triangle3D tr)
{

//	virtual bool intersect(Ray3D r);
//*
//bool IntersectTriangleBox (TRIANGLE * T, VECTOR Min, VECTOR Max)
//{
    Vector3D Qin, Qout, p1, p2;

    // test the triangle edges against the box
	if (intersectRayBox (vertexlist[tr.vertIdx[0]], vertexlist[tr.vertIdx[1]], &Qin, &Qout, 1))
        return true;
    if (intersectRayBox (vertexlist[tr.vertIdx[1]],vertexlist[tr.vertIdx[2]], &Qin, &Qout, 1))
        return true;
    if (intersectRayBox (vertexlist[tr.vertIdx[2]], vertexlist[tr.vertIdx[0]], &Qin, &Qout, 1))
        return true;

    // test the box edges against the triangle
    //CopyVector (p1, Min);
	p1 = min;
    //SetVector (p2, Min[X], Min[Y], Max[Z]);
	p2.x = min.x;
	p2.y = min.y;
	p2.z = max.z;
	//intersect(Ray3D r, Vector3D * vertexlist, Vector3D * normallist);
   // if (RayTriangleIntersection (p1, p2, tr, Qin, &t, LINE_SEG))
	Ray3D r(p1, p2-p1);
	if(tr.intersect(r, vertexlist))
        return true;

   // SetVector (p2, Min[X], Max[Y], Min[Z]);
	p2.x = min.x;
	p2.y = max.y;
	p2.z = min.z;
	r.dir = p2 - p1;
	if (tr.intersect(r, vertexlist))
        return true;

	//SetVector (p2, Max[X], Min[Y], Min[Z]);
	p2.x = max.x;
	p2.y = min.y;
	p2.z = min.z;
	r.dir = p2 - p1;
	if (tr.intersect(r, vertexlist))
        return true;

	r.origin = max;
	p2.x = max.x;
	p2.y = max.y;
	p2.z = min.z;
    //CopyVector (p1, Max);
    //SetVector (p2, Max[X], Max[Y], Min[Z]);
    r.dir = p2 - p1;
	if (tr.intersect(r, vertexlist))
		return true;

	//SetVector (p2, Max[X], Min[Y], Max[Z]);
	p2.x = max.x;
	p2.y = min.y;
	p2.z = max.z;
    r.dir = p2 - p1;
	if (tr.intersect(r, vertexlist))
        return true;

	//SetVector (p2, Min[X], Max[Y], Max[Z]);
	p2.x = min.x;
	p2.y = max.y;
	p2.z = max.z;
    r.dir = p2 - p1;
	if (tr.intersect(r, vertexlist))
        return true;

	//SetVector (p1, Min[X], Max[Y], Min[Z]);
    //SetVector (p2, Max[X], Max[Y], Min[Z]);
	p1.x = min.x;
	p1.y = max.y;
	p1.z = min.z;
	p2.x = max.x;
	p2.y = max.y;
	p2.z = min.z;
	r.origin = p1;
    r.dir = p2 - p1;
	if (tr.intersect(r, vertexlist))
        return true;

	//SetVector (p1, Max[X], Min[Y], Min[Z]);
    //SetVector (p2, Max[X], Max[Y], Min[Z]);

	p1.x = max.x;
	p1.y = min.y;
	p1.z = min.z;
	p2.x = max.x;
	p2.y = max.y;
	p2.z = min.z;
	r.origin = p1;
    r.dir = p2 - p1;
	if (tr.intersect(r, vertexlist))
		return true;

	//SetVector (p1, Max[X], Min[Y], Min[Z]);
    //SetVector (p2, Max[X], Min[Y], Max[Z]);

	p1.x = max.x;
	p1.y = min.y;
	p1.z = min.z;
	p2.x = max.x;
	p2.y = min.y;
	p2.z = max.z;
	r.origin = p1;
	r.dir = p2 - p1;
	if (tr.intersect(r, vertexlist))
        return true;
   // SetVector (p1, Min[X], Min[Y], Max[Z]);
    //SetVector (p2, Min[X], Max[Y], Max[Z]);

    p1.x = min.x;
	p1.y = min.y;
	p1.z = max.z;
	p2.x = min.x;
	p2.y = max.y;
	p2.z = max.z;
	r.origin = p1;
	r.dir = p2 - p1;
	if (tr.intersect(r, vertexlist))
        return true;

    //SetVector (p1, Min[X], Min[Y], Max[Z]);
    //SetVector (p2, Max[X], Min[Y], Max[Z]);
	p1.x = min.x;
	p1.y = min.y;
	p1.z = max.z;
	p2.x = max.x;
	p2.y = min.y;
	p2.z = max.z;
	r.origin = p1;
	r.dir = p2 - p1;
	if (tr.intersect(r, vertexlist))
        return true;
    return false;
}
//*/
//*
bool BoxBin::intersectRayBox (Vector3D Orig, Vector3D End, Vector3D * Qin, Vector3D * Qout, int IsSegment)
{
    DBL  t1x, t2x, t1y, t2y, t1z, t2z, tfar, tnear, dx, dy, dz, bminx,
           bminy, bminz, bmaxx, bmaxy, bmaxz, ox, oy, oz;

    ox = Orig.x;
    oy = Orig.y;
    oz = Orig.z;
    dx = End.x - ox;
    dy = End.y - oy;
    dz = End.z - oz;

    //fix the error in A. Watt implementation
    if (dx > 0)
    {
        bminx = min.x;
        bmaxx = max.x;
    }
    else
    {
        bminx = max.x;
        bmaxx = min.x;
    }
    if (dy > 0)
    {
        bminy = min.y;
        bmaxy = max.y;
    }
    else
    {
        bminy = max.y;
        bmaxy = min.y;
    }
    if (dz > 0)
    {
        bminz = min.z;
        bmaxz = max.z;
    }
    else
    {
        bminz = max.z;
        bmaxz = min.z;
    }
    // Watt bboxes with parallel rays support
    if (dx == 0.0)
    {
        if ((bminx - ox) * (bmaxx - ox) > 0)
            return false;
        t1x = -FLT_MAX;
        t2x = FLT_MAX;
    }
    else
    {
        t1x = (bminx - ox) / dx;
        t2x = (bmaxx - ox) / dx;
    }
    if (dy == 0.0)
    {
        if ((bminy - oy) * (bmaxy - oy) > 0)
            return false;
        t1y = -FLT_MAX;
        t2y = FLT_MAX;
    }
    else
    {
        t1y = (bminy - oy) / dy;
        t2y = (bmaxy - oy) / dy;
    }
    if (dz == 0.0)
    {
        if ((bminz - oz) * (bmaxz - oz) > 0)
            return false;
        t1z = -FLT_MAX;
        t2z = FLT_MAX;
    }
    else
    {
        t1z = (bminz - oz) / dz;
        t2z = (bmaxz - oz) / dz;
    }
#ifdef WIN32
    tfar = min (t2x, min (t2y, t2z));
    tnear = max (t1x, max (t1y, t1z));
#else
    tfar = fmin (t2x, fmin (t2y, t2z));
    tnear = fmax (t1x, fmax (t1y, t1z));
#endif
    if (tnear > tfar)
        return false;

    // Check the case of a line segment
    if (IsSegment * tnear > 1.0)
        return false;
    return true;
}
//*

