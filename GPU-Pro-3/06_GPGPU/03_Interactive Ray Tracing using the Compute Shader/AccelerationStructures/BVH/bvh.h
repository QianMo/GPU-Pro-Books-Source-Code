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

#ifndef BVH_H
#define BVH_H

// bvh.h
#include <stdlib.h>
#ifdef WINDOWS
#include <windows.h>
#elif defined(LINUX)
#include <DataTypes.h>
#endif
#include "Memory.h"
#include "BBox.h"
#include "Geometry.h"
#include "Ray.h"
#include "AccelerationStructure.h"

class BBox;

// BVH Local Declarations
struct BVHPrimitiveInfo 
{
    BVHPrimitiveInfo() {}
    BVHPrimitiveInfo(int pn, const BBox &b) : primitiveNumber(pn), bounds(b) 
	{
        centroid = .5f * b.pMin + .5f * b.pMax;
    }
    int primitiveNumber;
    Point centroid;
    BBox bounds;
};

struct BVHBuildNode 
{
    // BVHBuildNode Public Methods
    BVHBuildNode() { children[0] = children[1] = NULL; }
    void InitLeaf(uint32_t first, uint32_t n, const BBox &b) 
	{
        firstPrimOffset = first;
        nPrimitives = n;
        bounds = b;
    }
    void InitInterior(uint32_t axis, BVHBuildNode *c0, BVHBuildNode *c1) 
	{
        children[0] = c0;
        children[1] = c1;
        bounds = Union(c0->bounds, c1->bounds);
        splitAxis = axis;
        nPrimitives = 0;
    }
    BBox bounds;
    BVHBuildNode *children[2];
    uint32_t splitAxis, firstPrimOffset, nPrimitives;
};

struct CompareToMid 
{
    CompareToMid(int d, float m) { dim = d; mid = m; }
    int dim;
    float mid;
    bool operator()(const BVHPrimitiveInfo &a) const 
	{
        return a.centroid[dim] < mid;
    }
};

struct ComparePoints 
{
    ComparePoints(int d) { dim = d; }
    int dim;
    bool operator()(const BVHPrimitiveInfo &a, const BVHPrimitiveInfo &b) const 
	{
        return a.centroid[dim] < b.centroid[dim];
    }
};

struct LinearBVHNode 
{
    BBox bounds;
	int nPrimitives;	// 0 -> interior node
    union 
	{
        uint32_t primitivesOffset;    // leaf
        uint32_t secondChildOffset;   // interior
    };
};

static inline bool IntersectP(const BBox &bounds, const Ray &ray,const Vector &invDir, const uint32_t dirIsNeg[3]) 
{
    // Check for ray intersection against $x$ and $y$ slabs
    float tmin =  (bounds[  dirIsNeg[0]].x - ray.o.x) * invDir.x;
    float tmax =  (bounds[1-dirIsNeg[0]].x - ray.o.x) * invDir.x;
    float tymin = (bounds[  dirIsNeg[1]].y - ray.o.y) * invDir.y;
    float tymax = (bounds[1-dirIsNeg[1]].y - ray.o.y) * invDir.y;
    if ((tmin > tymax) || (tymin > tmax))
        return false;
    if (tymin > tmin) tmin = tymin;
    if (tymax < tmax) tmax = tymax;

    // Check for ray intersection against $z$ slab
    float tzmin = (bounds[  dirIsNeg[2]].z - ray.o.z) * invDir.z;
    float tzmax = (bounds[1-dirIsNeg[2]].z - ray.o.z) * invDir.z;
    if ((tmin > tzmax) || (tzmin > tmax))
        return false;

    if (tzmin > tmin)
        tmin = tzmin;
    if (tzmax < tmax)
        tmax = tzmax;
			
    return (tmin < ray.maxt) && (tmax > ray.mint);
}

enum SplitMethod { SPLIT_MIDDLE, SPLIT_EQUAL_COUNTS, SPLIT_SAH };

// BVH Declarations
class BVH : public AccelerationStructure
{
public:
    // BVH Public Methods
    BVH(Primitive** a_Primitive = NULL, unsigned int a_NumPrimitives = 0, uint32_t maxPrims = 1, const string &sm = "sah");
	~BVH();

	// BVH Functions
	TIntersection		IntersectP(Ray &a_Ray);
	void				PrintOutput(float &time);
	void				Build();
	BBox				WorldBound() const { return nodes ? nodes[0].bounds : BBox(); } 

	// Getters
	LinearBVHNode*		GetNodes( void ) { return nodes; }
	BVHPrimitiveInfo*	GetPrimitives( void ) { return primitives; }
	unsigned int		GetNumberOfElements() { return totalNodes; }

private:
    // BVH Private Methods
	uint32_t			FlattenBVHTree(BVHBuildNode *node, uint32_t *offset);
    BVHBuildNode*		RecursiveBuild(MemoryArena &buildArena,
									BVHPrimitiveInfo *buildData, uint32_t start, uint32_t end,
									uint32_t *totalNodes, BVHPrimitiveInfo* orderedPrims);
    // BVH Private Data
    uint32_t			maxPrimsInNode;
	uint32_t			totalNodes;
	SplitMethod			splitMethod;
	BVHPrimitiveInfo	*primitives;
	LinearBVHNode		*nodes;
};

#endif // BVH_H