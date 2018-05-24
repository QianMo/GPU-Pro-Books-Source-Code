#ifndef __MESHFIL_H__
#define __MESHFIL_H__

#include "DXUT.h"

#define DISABLE_QUAT

#include <geommath/geommath.h>

struct SFilVert
{
	Vec3 vert;	// position
	float s, t;	// texcoord

	Vec3 vOs; float fMagS;	// first order derivative with respect to, s
	Vec3 vOt; float fMagT;	// first order derivative with respect to, t
	Vec3 norm;				// normal
};

class CQuadTree;



class CMeshFil
{
public:
	bool ReadMeshFil(ID3D11Device* pd3dDev, const char file_name[], const float fScale = 1.0f, const bool bCenter = true, const bool bGenQuadTree = false);
	void CleanUp();
	int GetNrTriangles() { return m_iNrFaces; }
	ID3D11Buffer *const * GetVertexBuffer() { return &m_pVertStream; }
	ID3D11Buffer * GetIndexBuffer() { return m_pIndexStream; }
	const Vec3 GetMin() const { return m_vMin; }
	const Vec3 GetMax() const { return m_vMax; }
	float QueryTopY(const float fX, const float fZ) const;


	CMeshFil();
	~CMeshFil();


private:
	int m_iNrVerts;
	int m_iNrFaces;
	SFilVert * m_vVerts;
	int * m_iIndices;

	Vec3 m_vMin, m_vMax;

	CQuadTree * m_pQuadTree;

private:	// d3d specific
	ID3D11Buffer * m_pVertStream;
	ID3D11Buffer * m_pIndexStream;
};

#endif