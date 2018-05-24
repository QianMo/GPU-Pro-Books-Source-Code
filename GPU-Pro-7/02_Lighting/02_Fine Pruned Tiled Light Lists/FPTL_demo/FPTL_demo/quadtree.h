#ifndef __QUADTREE_H__
#define __QUADTREE_H__

#define DISABLE_QUAT

#include <geommath/geommath.h>
#include <stdio.h>

class CQuadTree
{
public:
	float QueryTopY(const float fX, const float fZ) const;
	bool InitTree(const int iMaxNrTriangles);
	void AddTriangle(const Vec3 &p0, const Vec3 &p1, const Vec3 &p2);
	bool BuildTree();

	CQuadTree();
	~CQuadTree();

private:
	struct SQuadNode
	{
		float fMiX, fMaX;
		float fMiZ, fMaZ;

		SQuadNode * ChildNodes[4];
		int * piTriList;		// for leaf nodes
		int iTriCount;			// for leaf nodes

		SQuadNode()
		{
			for(int i=0; i<4; i++) ChildNodes[i]=NULL;
			piTriList=NULL; iTriCount=0;
		}
	};

	void RecurCleanUp(SQuadNode * pNode);
	bool RecursivGenNode(SQuadNode ** ppGenNode, const int piTriList[], const int iNrTrisIn, const int iDepth,
							const float fMiX, const float fMaX, const float fMiZ, const float fMaZ);
	void CleanUp();
	bool CheckTriInside(const int iTriIndex, const float fMiX, const float fMaX, const float fMiZ, const float fMaZ);

	SQuadNode * m_pTopNode;

	Vec3 * m_pvTriangles;
	int m_iNrTotTriangles;
	int m_iAddedTriangles;
	bool m_bTreeWasBuilt;
};


#endif