#include "quadtree.h"
#include <assert.h>

float CQuadTree::QueryTopY(const float fX, const float fZ) const
{
	float fRes = -10000000000.0f;

	SQuadNode * pNode = m_pTopNode;
	while(pNode!=NULL && pNode->piTriList==NULL)
	{
		float fAvgX = 0.5f*(pNode->fMiX+pNode->fMaX);
		float fAvgZ = 0.5f*(pNode->fMiZ+pNode->fMaZ);

		int index = (fX<fAvgX ? 0 : 1) + (fZ<fAvgZ ? 0 : 2);
		pNode = pNode->ChildNodes[index];
	}

	assert(pNode->piTriList!=NULL);
	if(pNode->piTriList!=NULL)
	{
		bool bNotSetYet = true;
		for(int i=0; i<pNode->iTriCount; i++)
		{
			const int idx = pNode->piTriList[i];
			const Vec3 &p0 = m_pvTriangles[3*idx+0];
			const Vec3 &p1 = m_pvTriangles[3*idx+1];
			const Vec3 &p2 = m_pvTriangles[3*idx+2];

			const float fSign0 = (p1.x-p0.x)*(p0.z-fZ) - (p1.z-p0.z)*(p0.x-fX);
			const float fSign1 = (p2.x-p1.x)*(p1.z-fZ) - (p2.z-p1.z)*(p1.x-fX);
			const float fSign2 = (p0.x-p2.x)*(p2.z-fZ) - (p0.z-p2.z)*(p2.x-fX);

			if( (fSign0<=0 && fSign1<=0 && fSign2<=0) || (fSign0>=0 && fSign1>=0 && fSign2>=0) )
			{
				// (p0.x-p2.x)*s + (p1.x-p2.x)*t = fX-p2.x
				// (p0.z-p2.z)*s + (p1.z-p2.z)*t = fZ-p2.z

				const float fA = p0.x-p2.x, fB = p1.x-p2.x;
				const float fC = p0.z-p2.z, fD = p1.z-p2.z;
				const float fK0 = fX-p2.x, fK1 = fZ-p2.z;

				double fDet = ((double) fA)*fD - ((double) fC)*fB;
				if(fDet!=0)
				{
					const float fS = (fD*fK0 - fB*fK1)/fDet;
					const float fT = ((-fC)*fK0 + fA*fK1)/fDet;

					const float fY = fS*p0.y + fT*p1.y + (1-fS-fT)*p2.y;
					if(bNotSetYet) { fRes=fY; bNotSetYet=false; }
					else if(fY>fRes) fRes=fY;
				}
			}
		}
	}

	return fRes;
}


bool CQuadTree::InitTree(const int iMaxNrTriangles)
{
	CleanUp();

	bool bRes = false;

	m_pvTriangles=new Vec3[iMaxNrTriangles*3];
	if(m_pvTriangles!=NULL)
	{
		m_iNrTotTriangles=iMaxNrTriangles;
		bRes = true;
	}

	return bRes;
}

void CQuadTree::AddTriangle(const Vec3 &p0, const Vec3 &p1, const Vec3 &p2)
{
	if(m_iNrTotTriangles>0 && m_iAddedTriangles<m_iNrTotTriangles)
	{
		m_pvTriangles[3*m_iAddedTriangles+0] = p0;
		m_pvTriangles[3*m_iAddedTriangles+1] = p1;
		m_pvTriangles[3*m_iAddedTriangles+2] = p2;

		++m_iAddedTriangles;
	}
}

bool CQuadTree::BuildTree()
{
	if(m_iNrTotTriangles>0 && m_iAddedTriangles==m_iNrTotTriangles)
	{
		float fMiX, fMaX, fMiZ, fMaZ;
		fMiX = m_pvTriangles[0].x; fMaX = fMiX;
		fMiZ = m_pvTriangles[0].z; fMaZ = fMiZ;
		for(int i=1; i<(m_iNrTotTriangles*3); i++)
		{
			if(fMiX>m_pvTriangles[i].x) fMiX=m_pvTriangles[i].x;
			else if(fMaX<m_pvTriangles[i].x) fMaX=m_pvTriangles[i].x;
			if(fMiZ>m_pvTriangles[i].z) fMiZ=m_pvTriangles[i].z;
			else if(fMaZ<m_pvTriangles[i].z) fMaZ=m_pvTriangles[i].z;
		}

		int * piTris = new int[m_iNrTotTriangles];
		if(piTris!=NULL)
		{
			for(int t=0; t<m_iNrTotTriangles; t++) 
				piTris[t]=t;
	
			m_bTreeWasBuilt = RecursivGenNode(&m_pTopNode, piTris, m_iNrTotTriangles, 0, fMiX, fMaX, fMiZ, fMaZ);
			delete [] piTris; piTris=NULL;
		}
		if(!m_bTreeWasBuilt) CleanUp();
	}

	return m_bTreeWasBuilt;
}

bool CQuadTree::RecursivGenNode(SQuadNode ** ppGenNode, const int piTriList[], const int iNrTrisIn, const int iDepth,
								const float fMiX, const float fMaX, const float fMiZ, const float fMaZ)
{
	if(iNrTrisIn==0) { *ppGenNode=NULL; return true; }

	bool bRes = false;
	SQuadNode * pNode = new SQuadNode;
	if(pNode!=NULL)
	{
		pNode->fMiX = fMiX; pNode->fMaX = fMaX;
		pNode->fMiZ = fMiZ; pNode->fMaZ = fMaZ;

		if(iNrTrisIn<12 || iDepth>15)		// gen leaf node
		{
			int * piTrisOut = new int[iNrTrisIn];
			if(piTrisOut!=NULL)
			{
				for(int t=0; t<iNrTrisIn; t++) piTrisOut[t]=piTriList[t];
				pNode->piTriList = piTrisOut;
				pNode->iTriCount = iNrTrisIn;

				bRes=true;
			}
		}
		else
		{
			int * piTrisOut = new int[iNrTrisIn];
			if(piTrisOut!=NULL)
			{
				float fAvgX = 0.5f*(fMiX+fMaX);
				float fAvgZ = 0.5f*(fMiZ+fMaZ);

				int c=0;
				bool bAllGood = true;
				while(c<4 && bAllGood)
				{
					const float fMiX_2 = (c&1)==0 ? fMiX : fAvgX;
					const float fMaX_2 = (c&1)==0 ? fAvgX : fMaX;
					const float fMiZ_2 = (c&2)==0 ? fMiZ : fAvgZ;
					const float fMaZ_2 = (c&2)==0 ? fAvgZ : fMaZ;

					int iNrTrisOut = 0;
					for(int t=0; t<iNrTrisIn; t++)
					{
						if( CheckTriInside(piTriList[t], fMiX_2, fMaX_2, fMiZ_2, fMaZ_2) )
							piTrisOut[iNrTrisOut++] = piTriList[t];
					}
					bAllGood &= RecursivGenNode(&pNode->ChildNodes[c], piTrisOut, iNrTrisOut, iDepth+1, fMiX_2, fMaX_2, fMiZ_2, fMaZ_2);

					if(bAllGood) ++c;
				}

				bRes = bAllGood;
			}

			if(piTrisOut!=NULL) { delete [] piTrisOut; piTrisOut=NULL; }
		}
	}

	// we insert it no matter what to make sure the node gets cleaned up
	// if tree generation fails.
	*ppGenNode = pNode;

	return bRes;
}

bool CQuadTree::CheckTriInside(const int iTriIndex, const float fMiX, const float fMaX, const float fMiZ, const float fMaZ)
{
	const Vec3 &p0 = m_pvTriangles[3*iTriIndex+0];
	const Vec3 &p1 = m_pvTriangles[3*iTriIndex+1];
	const Vec3 &p2 = m_pvTriangles[3*iTriIndex+2];

	const float miX = (p0.x<=p1.x && p0.x<=p2.x) ? p0.x : (p1.x<=p2.x ? p1.x : p2.x);
	const float maX = (p0.x>=p1.x && p0.x>=p2.x) ? p0.x : (p1.x>=p2.x ? p1.x : p2.x);
	const float miZ = (p0.z<=p1.z && p0.z<=p2.z) ? p0.z : (p1.z<=p2.z ? p1.z : p2.z);
	const float maZ = (p0.z>=p1.z && p0.z>=p2.z) ? p0.z : (p1.z>=p2.z ? p1.z : p2.z);

	if( (maX>fMiX && maZ>fMiZ) && (miX<fMaX && miZ<fMaZ) )
		return true;
	else
		return false;
}


void CQuadTree::CleanUp()
{
	RecurCleanUp(m_pTopNode);

	if(m_pvTriangles!=NULL) { delete [] m_pvTriangles; m_pvTriangles=NULL; }

	m_iNrTotTriangles=0;
	m_iAddedTriangles=0;
	m_bTreeWasBuilt=false;
}


void CQuadTree::RecurCleanUp(SQuadNode * pNode)
{
	if(pNode==NULL) return; 
	for(int i=0; i<4; i++) RecurCleanUp(pNode->ChildNodes[i]);
	if(pNode->piTriList!=NULL) delete [] pNode->piTriList;
	delete [] pNode;
}

CQuadTree::CQuadTree()
{
	m_pvTriangles=NULL;
	m_iNrTotTriangles=0;
	m_bTreeWasBuilt=false;
	m_pTopNode=NULL;
}

CQuadTree::~CQuadTree()
{

}