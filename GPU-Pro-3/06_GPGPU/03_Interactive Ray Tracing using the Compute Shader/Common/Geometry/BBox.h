/*
 * pbrt source code Copyright(c) 1998-2004 Matt Pharr and Greg Humphreys
 *
 * All Rights Reserved.
 * For educational use only; commercial use expressly forbidden.
 * NO WARRANTY, express or implied, for this software.
 * (See file License.txt for complete license)
 */

// --------------------------------------------------------------------	//
// This code was modified by the authors of the demo. The original		//
// PBRT code is available at https://github.com/mmp/pbrt-v2. Basically, //
// we removed all STL-based implementation and it was merged with		//
// our current framework.												//
// --------------------------------------------------------------------	//

#ifndef __BBOX_H__
#define __BBOX_H__

#include "Geometry.h"
#include "Ray.h"

class BBox 
{
public:
    // BBox Public Methods
    BBox() {
        pMin = Point( INFINITY,  INFINITY,  INFINITY);
        pMax = Point(-INFINITY, -INFINITY, -INFINITY);
    }
    BBox(const Point &p) : pMin(p), pMax(p) { }
    BBox(const Point &p1, const Point &p2) {
        pMin = Point(min(p1.x, p2.x), min(p1.y, p2.y), min(p1.z, p2.z));
        pMax = Point(max(p1.x, p2.x), max(p1.y, p2.y), max(p1.z, p2.z));
    }
    //friend BBox Union(const BBox &b, const Point &p);
    //friend BBox Union(const BBox &b, const BBox &b2);
    bool Overlaps(const BBox &b) const {
        bool x = (pMax.x >= b.pMin.x) && (pMin.x <= b.pMax.x);
        bool y = (pMax.y >= b.pMin.y) && (pMin.y <= b.pMax.y);
        bool z = (pMax.z >= b.pMin.z) && (pMin.z <= b.pMax.z);
        return (x && y && z);
    }
    bool Inside(const Point &pt) const {
        return (pt.x >= pMin.x && pt.x <= pMax.x &&
                pt.y >= pMin.y && pt.y <= pMax.y &&
                pt.z >= pMin.z && pt.z <= pMax.z);
    }
    void Expand(float delta) {
        pMin -= Vector(delta, delta, delta);
        pMax += Vector(delta, delta, delta);
    }
    float SurfaceArea() const {
        Vector d = pMax - pMin;
        return 2.f * (d.x * d.y + d.x * d.z + d.y * d.z);
    }
    double Volume() const {
        Vector d = pMax - pMin;
        return d.x * d.y * d.z;
    }
    int MaximumExtent() const {
        Vector diag = pMax - pMin;
        if (diag.x > diag.y && diag.x > diag.z)
            return 0;
        else if (diag.y > diag.z)
            return 1;
        else
            return 2;
    }
	int MinimumExtent() const {
        Vector diag = pMax - pMin;
        if (diag.x < diag.y && diag.x < diag.z)
            return 0;
        else if (diag.y < diag.z)
            return 1;
        else
            return 2;
    }

    const Point &operator[](int i) const;
    Point &operator[](int i);
    Point Lerp(float tx, float ty, float tz) const {
        return Point(::Lerp(tx, pMin.x, pMax.x), ::Lerp(ty, pMin.y, pMax.y),
                     ::Lerp(tz, pMin.z, pMax.z));
    }
    Vector Offset(const Point &p) const {
        return Vector((p.x - pMin.x) / (pMax.x - pMin.x),
                      (p.y - pMin.y) / (pMax.y - pMin.y),
                      (p.z - pMin.z) / (pMax.z - pMin.z));
    }
    void BoundingSphere(Point *c, float *rad) const
	{
		//*c = .5f * pMin + .5f * pMax;
		//*rad = Inside(*c) ? Distance(*c, pMax) : 0.f;
	}

    bool IntersectP(const Ray &ray, float *hitt0 = NULL, float *hitt1 = NULL) const
	{
		float t0 = ray.mint, t1 = ray.maxt;
		for (int i = 0; i < 3; ++i) {
			// Update interval for _i_th bounding box slab
			float invRayDir = 1.f / ray.d[i];
			float tNear = (pMin[i] - ray.o[i]) * invRayDir;
			float tFar  = (pMax[i] - ray.o[i]) * invRayDir;

			// Update parametric interval from slab intersection $t$s
			if (tNear > tFar) swap(tNear, tFar);
			t0 = tNear > t0 ? tNear : t0;
			t1 = tFar  < t1 ? tFar  : t1;
			if (t0 > t1) return false;
		}
		if (hitt0) *hitt0 = t0;
		if (hitt1) *hitt1 = t1;
		return true;
	}

	// BBox Method Definitions
	friend BBox Union(const BBox &b, const Point &p) {
		BBox ret = b;
		ret.pMin.x = min(b.pMin.x, p.x);
		ret.pMin.y = min(b.pMin.y, p.y);
		ret.pMin.z = min(b.pMin.z, p.z);
		ret.pMax.x = max(b.pMax.x, p.x);
		ret.pMax.y = max(b.pMax.y, p.y);
		ret.pMax.z = max(b.pMax.z, p.z);
		return ret;
	}


	friend BBox Union(const BBox &b, const BBox &b2) {
		BBox ret;
		ret.pMin.x = min(b.pMin.x, b2.pMin.x);
		ret.pMin.y = min(b.pMin.y, b2.pMin.y);
		ret.pMin.z = min(b.pMin.z, b2.pMin.z);
		ret.pMax.x = max(b.pMax.x, b2.pMax.x);
		ret.pMax.y = max(b.pMax.y, b2.pMax.y);
		ret.pMax.z = max(b.pMax.z, b2.pMax.z);
		return ret;
	}

    // BBox Public Data
    Point pMin, pMax;
};

inline const Point &BBox::operator[](int i) const 
{
    Assert(i == 0 || i == 1);
    return (&pMin)[i];
}

inline Point &BBox::operator[](int i) 
{
    Assert(i == 0 || i == 1);
    return (&pMin)[i];
}

#endif