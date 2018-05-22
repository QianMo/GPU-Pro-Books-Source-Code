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

#ifndef __RAY_H__
#define __RAY_H__

#include "Geometry.h"

class Ray 
{
public:
    // Ray Public Methods
    Ray() : mint(0.f), maxt(INFINITY), TriangleID(-1), factor(1.f,1.f,1.f) { }
    Ray(const Point &origin, const Vector &direction,
        float start, float end = INFINITY, float t = 0.f, int d = 0)
        : o(origin), d(direction), mint(start), maxt(end), TriangleID(-1) { }
    Ray(const Point &origin, const Vector &direction, const Ray &parent,
        float start, float end = INFINITY)
        : o(origin), d(direction), mint(start), maxt(end),
         TriangleID(-1) { }
    Point operator()(float t) const { return o + d * t; }
    bool HasNaNs() const {
        return (o.HasNaNs() || d.HasNaNs() ||
                isnan(mint) || isnan(maxt));
    }

    // Ray Public Data
    Point o;
    Vector d;
	Vector factor;

    mutable float mint, maxt;
    //float time;
    //int depth;
	int TriangleID;
};

#endif