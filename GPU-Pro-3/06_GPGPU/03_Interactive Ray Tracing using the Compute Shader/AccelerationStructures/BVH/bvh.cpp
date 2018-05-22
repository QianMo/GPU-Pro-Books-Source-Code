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

#include "bvh.h"

#ifdef LINUX
#include <sys/time.h>
#endif

struct CompareToBucket 
{
    CompareToBucket(int split, int num, int d, const BBox &b)
        : centroidBounds(b)
    { splitBucket = split; nBuckets = num; dim = d; }
    bool operator()(const BVHPrimitiveInfo &p) const;

    int splitBucket;
    int nBuckets, dim;
    const BBox &centroidBounds;
};

bool CompareToBucket::operator()(const BVHPrimitiveInfo &p) const 
{
    int b = nBuckets * ((p.centroid[dim] - centroidBounds.pMin[dim]) /
            (centroidBounds.pMax[dim] - centroidBounds.pMin[dim]));
    if (b == nBuckets) b = nBuckets-1;
    Assert(b >= 0 && b < nBuckets);
    return b <= splitBucket;
}

// BVH Method Definitions
BVH::BVH(Primitive** pPrimitives, unsigned int uiNumPrimitives, uint32_t mp, const string &sm) 
{
	m_pPrimitives = pPrimitives;
	m_uiNumPrimitives = uiNumPrimitives;
	m_sName = "BVH";
    maxPrimsInNode = min(255u, mp);

    if (sm == "sah")         splitMethod = SPLIT_SAH;
    else if (sm == "middle") splitMethod = SPLIT_MIDDLE;
    else if (sm == "equal")  splitMethod = SPLIT_EQUAL_COUNTS;
    else splitMethod = SPLIT_SAH;
}

BVH::~BVH() 
{
	delete [] primitives;
    FreeAligned(nodes);
}

void BVH::Build()
{
	printf("Building BVH...\n");

	primitives = new BVHPrimitiveInfo[m_uiNumPrimitives];
	BVHPrimitiveInfo *buildData = new BVHPrimitiveInfo[m_uiNumPrimitives];

	for (unsigned int i = 0; i < m_uiNumPrimitives; ++i) 
	{ 
		Point Vertex1(m_pPrimitives[i]->GetVertex(0)->Pos);
		Point Vertex2(m_pPrimitives[i]->GetVertex(1)->Pos);
		Point Vertex3(m_pPrimitives[i]->GetVertex(2)->Pos);
		BBox bbox = Union(BBox(Vertex1, Vertex2),Vertex3);

		primitives[i] = BVHPrimitiveInfo(i, bbox);
		buildData[i] = BVHPrimitiveInfo(i, bbox);
    }

    // Recursively build BVH tree for primitives
    MemoryArena buildArena;
    totalNodes = 0;
	BVHPrimitiveInfo*  orderedPrims = new BVHPrimitiveInfo[m_uiNumPrimitives];
    BVHBuildNode *root = RecursiveBuild(buildArena, buildData, 0, m_uiNumPrimitives, &totalNodes, orderedPrims);

	swap(primitives, orderedPrims);
    
	// Compute representation of depth-first traversal of BVH tree
    nodes = AllocAligned<LinearBVHNode>(totalNodes);
    for (uint32_t i = 0; i < totalNodes; ++i)
	{
        new (&nodes[i]) LinearBVHNode;
	}
    uint32_t offset = 0;
    FlattenBVHTree(root, &offset);
    Assert(offset == totalNodes);

	delete[] orderedPrims;
	delete[] buildData;
}

BVHBuildNode *BVH::RecursiveBuild(MemoryArena &buildArena,
        BVHPrimitiveInfo *buildData, uint32_t start,
        uint32_t end, uint32_t *totalNodes,
        BVHPrimitiveInfo *orderedPrims) {
    Assert(start != end);
    (*totalNodes)++;
    BVHBuildNode *node = buildArena.Alloc<BVHBuildNode>();
    
	// Compute bounds of all primitives in BVH node
    BBox bbox;
	float minCost = 0.0f;
	
    for (uint32_t i = start; i < end; ++i)
        bbox = Union(bbox, buildData[i].bounds);
    uint32_t nPrimitives = end - start;
    if (nPrimitives == maxPrimsInNode) 
	{
        // Create leaf _BVHBuildNode_
		uint32_t firstPrimOffset = start;
        for (uint32_t i = start; i < end; ++i) {
            uint32_t primNum = buildData[i].primitiveNumber;
			orderedPrims[i] = primitives[primNum];
        }
        node->InitLeaf(firstPrimOffset, nPrimitives, bbox);
    }
    else 
	{
        // Compute bound of primitive centroids, choose split dimension _dim_
        BBox centroidBounds;
        for (uint32_t i = start; i < end; ++i)
            centroidBounds = Union(centroidBounds, buildData[i].centroid);
        int dim = centroidBounds.MaximumExtent();

        // Partition primitives into two sets and build children
        uint32_t mid = (start + end) / 2;
        if (maxPrimsInNode > 1 && centroidBounds.pMax[dim] == centroidBounds.pMin[dim]) 
		{
            // Create leaf _BVHBuildNode_
			uint32_t firstPrimOffset = start;
            for (uint32_t i = start; i < end; ++i) {
                uint32_t primNum = buildData[i].primitiveNumber;
				orderedPrims[i] = primitives[primNum];
            }
            node->InitLeaf(firstPrimOffset, nPrimitives, bbox);
            return node;
        } // END IF CENTROIDS

        // Partition primitives based on _splitMethod_
        switch (splitMethod) {
        case SPLIT_MIDDLE: {
            // Partition primitives through node's midpoint
            float pmid = .5f * (centroidBounds.pMin[dim] + centroidBounds.pMax[dim]);
            BVHPrimitiveInfo *midPtr = std::partition(&buildData[start],
                                                      &buildData[end-1]+1,
                                                      CompareToMid(dim, pmid));
            mid = midPtr - &buildData[0];
            break;
        }
        case SPLIT_EQUAL_COUNTS: {
            // Partition primitives into equally-sized subsets
            mid = (start + end) / 2;
            std::nth_element(&buildData[start], &buildData[mid],
                             &buildData[end-1]+1, ComparePoints(dim));
            break;
        }
        case SPLIT_SAH: default: 
		{
            // Partition primitives using approximate SAH
            if (nPrimitives <= 4) 
			{
                // Partition primitives into equally-sized subsets
                mid = (start + end) / 2;
                std::nth_element(&buildData[start], &buildData[mid],
                                 &buildData[end-1]+1, ComparePoints(dim));
            }
            else 
			{
				float invSAH = 1/bbox.SurfaceArea();

                // Allocate _BucketInfo_ for SAH partition buckets
                const int nBuckets = 12;
                struct BucketInfo {
                    BucketInfo() { count = 0; }
                    int count;
                    BBox bounds;
                };
                BucketInfo buckets[nBuckets];

                // Initialize _BucketInfo_ for SAH partition buckets
                for (uint32_t i = start; i < end; ++i) 
				{
                    int b = nBuckets *
                        ((buildData[i].centroid[dim] - centroidBounds.pMin[dim]) /
                         (centroidBounds.pMax[dim] - centroidBounds.pMin[dim]));
                    if (b == nBuckets) b = nBuckets-1;
                    Assert(b >= 0 && b < nBuckets);
                    buckets[b].count++;
                    buckets[b].bounds = Union(buckets[b].bounds, buildData[i].bounds);
                }

                // Compute costs for splitting after each bucket
                float cost[nBuckets-1];
                for (int i = 0; i < nBuckets-1; ++i) 
				{
                    BBox b0, b1;
                    int count0 = 0, count1 = 0;
                    for (int j = 0; j <= i; ++j) 
					{
                        b0 = Union(b0, buckets[j].bounds);
                        count0 += buckets[j].count;
                    }
                    for (int j = i+1; j < nBuckets; ++j) 
					{
                        b1 = Union(b1, buckets[j].bounds);
                        count1 += buckets[j].count;
                    }
                    cost[i] = .125f + (count0 * b0.SurfaceArea() + count1 * b1.SurfaceArea()) * invSAH;
                }

                // Find bucket to split at that minimizes SAH metric
                minCost = cost[0];
                uint32_t minCostSplit = 0;
                for (int i = 1; i < nBuckets-1; ++i) 
				{
                    if (cost[i] < minCost) 
					{
                        minCost = cost[i];
                        minCostSplit = i;
                    }
                }

                // Either create leaf or split primitives at selected SAH bucket
                if (nPrimitives > maxPrimsInNode ||
                    minCost < nPrimitives) 
				{
					// split primitives
                    BVHPrimitiveInfo *pmid = std::partition(&buildData[start],
                        &buildData[end-1]+1,
                        CompareToBucket(minCostSplit, nBuckets, dim, centroidBounds));
                    mid = pmid - &buildData[0];
                }
                else 
				{
                    // Create leaf _BVHBuildNode_
					uint32_t firstPrimOffset = start;
                    for (uint32_t i = start; i < end; ++i) 
					{
                        uint32_t primNum = buildData[i].primitiveNumber;
						orderedPrims[i] = primitives[primNum];
                    }
                    node->InitLeaf(firstPrimOffset, nPrimitives, bbox);
                }
            } // END ELSE SAH
            break;
        } // END CASE SAH
        } // END SWITCH
        node->InitInterior(dim,
                           RecursiveBuild(buildArena, buildData, start, mid,
                                          totalNodes, orderedPrims),
                           RecursiveBuild(buildArena, buildData, mid, end,
                                          totalNodes, orderedPrims)
						   );
    } // END ELSE
    return node;
}


uint32_t BVH::FlattenBVHTree(BVHBuildNode *node, uint32_t *offset) {
    LinearBVHNode *linearNode = &nodes[*offset];
    linearNode->bounds = node->bounds;
    uint32_t myOffset = (*offset)++;
    if (node->nPrimitives > 0) {
        Assert(!node->children[0] && !node->children[1]);
        linearNode->primitivesOffset = node->firstPrimOffset;
        linearNode->nPrimitives = node->nPrimitives;
    }
    else {
        // Creater interior flattened BVH node
		linearNode->nPrimitives = -static_cast<int>(node->splitAxis);
        FlattenBVHTree(node->children[0], offset);
        linearNode->secondChildOffset = FlattenBVHTree(node->children[1], offset);
    }

	return myOffset;
}

TIntersection BVH::IntersectP(Ray &a_Ray)
{
    if (!nodes) return false;

	TIntersection Intersect(-1,a_Ray.maxt);	

    Vector invDir(1.f / a_Ray.d.x, 1.f / a_Ray.d.y, 1.f / a_Ray.d.z);
    uint32_t dirIsNeg[3] = { invDir.x < 0, invDir.y < 0, invDir.z < 0 };
    uint32_t todo[64];
    uint32_t todoOffset = 0, nodeNum = 0;
    int TriangleID; 
	TIntersection test;	
	
	 while (true) 
	 {
		 const LinearBVHNode *node = &nodes[nodeNum];
        if (::IntersectP(node->bounds, a_Ray, invDir, dirIsNeg)) {
            // Process BVH node _node_ for traversal
			//m_Candidates++;
            if (node->nPrimitives > 0) {
                for (int i = 0; i < node->nPrimitives; ++i) {
					//m_Candidates++;
					if (primitives[node->primitivesOffset+i].bounds.IntersectP(a_Ray)){
						TriangleID = primitives[node->primitivesOffset+i].primitiveNumber;
						//candidates++;
						if (RayTriangleTest(a_Ray.o,a_Ray.d,test,TriangleID,m_pPrimitives) && test.t<Intersect.t) {
							Intersect = test;
							Intersect.IDTr = TriangleID;
							a_Ray.maxt = Intersect.t;
						}
					}
				}
                if (todoOffset == 0) break;
                nodeNum = todo[--todoOffset];
            }
			else {
				if (dirIsNeg[-node->nPrimitives]) {
					/// second child first
					todo[todoOffset++] = nodeNum + 1;
					nodeNum = node->secondChildOffset;
				}
				else {
                todo[todoOffset++] = node->secondChildOffset;
                nodeNum = nodeNum + 1;
				}
			}
        }
        else {
            if (todoOffset == 0) break;
            nodeNum = todo[--todoOffset];
        }
     }
	 return Intersect;
}

void BVH::PrintOutput(float &tiempototal)
{
	printf("BVH: %d Nodes * %d Bytes per Node = %.2f MB \n",totalNodes, sizeof(LinearBVHNode), (float)(totalNodes*sizeof(LinearBVHNode))/(float)(1024*1024));
}