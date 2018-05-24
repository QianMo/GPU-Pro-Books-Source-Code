#include "mesh_fil.h"
#include "quadtree.h"
#include <stdio.h>


bool CMeshFil::ReadMeshFil(ID3D11Device* pd3dDev, const char file_name[], const float fScale, const bool bCenter, const bool bGenQuadTree)
{
	// clean up previous
	CleanUp();


	bool bSuccess = false;
	FILE * fptr = fopen(file_name, "rb");
	if(fptr!=NULL)
	{
		fread((void *) &m_iNrVerts, sizeof(int), 1, fptr);
		fread((void *)&m_iNrFaces, sizeof(int), 1, fptr);
		m_vVerts = new SFilVert[m_iNrVerts];
		m_iIndices = new int[3*m_iNrFaces];

		if(m_vVerts!=NULL && m_iIndices!=NULL)
		{
			bSuccess=true;

			fread(m_vVerts, sizeof(SFilVert), m_iNrVerts, fptr);
			fread(m_iIndices, sizeof(int), 3*m_iNrFaces, fptr);
			fclose(fptr);

			// flip faces
			for(int q=0; q<m_iNrFaces; q++)
			{
				int index = m_iIndices[q*3+1];
				m_iIndices[q*3+1] = m_iIndices[q*3+2];
				m_iIndices[q*3+2] = index;
			}

			// scale and center
			Vec3 vMin=m_vVerts[0].vert * fScale;
			Vec3 vMax=vMin;
			for(int k=0; k<m_iNrVerts; k++)
			{
				m_vVerts[k].vert *= fScale;
				m_vVerts[k].fMagS *= fScale;
				m_vVerts[k].fMagT *= fScale;
				if(vMin.x>m_vVerts[k].vert.x) vMin.x=m_vVerts[k].vert.x;
				else if(vMax.x<m_vVerts[k].vert.x) vMax.x=m_vVerts[k].vert.x;
				if(vMin.y>m_vVerts[k].vert.y) vMin.y=m_vVerts[k].vert.y;
				else if(vMax.y<m_vVerts[k].vert.y) vMax.y=m_vVerts[k].vert.y;
				if(vMin.z>m_vVerts[k].vert.z) vMin.z=m_vVerts[k].vert.z;
				else if(vMax.z<m_vVerts[k].vert.z) vMax.z=m_vVerts[k].vert.z;
			}
			const Vec3 vCen = 0.5f*(vMax+vMin);

			if(bCenter)
			{
				for(int k=0; k<m_iNrVerts; k++)
				{
					m_vVerts[k].vert -= vCen;
				}
				vMin -= vCen; vMax -= vCen;
			}
			m_vMin = vMin; m_vMax = vMax;

			HRESULT hr;

			// Set initial data info
			D3D11_SUBRESOURCE_DATA InitData;
			InitData.pSysMem = m_vVerts;

			// Fill DX11 vertex buffer description
			D3D11_BUFFER_DESC     bd;
			bd.Usage =            D3D11_USAGE_DEFAULT;
			bd.ByteWidth =        sizeof( SFilVert ) * m_iNrVerts;
			bd.BindFlags =        D3D11_BIND_VERTEX_BUFFER;
			bd.CPUAccessFlags =   0;
			bd.MiscFlags =        0;

			// Create DX11 vertex buffer specifying initial data
			V( pd3dDev->CreateBuffer(&bd, &InitData, &m_pVertStream) );


			// Set initial data info
			InitData.pSysMem = m_iIndices;

			// Fill DX11 vertex buffer description
			bd.ByteWidth = sizeof(unsigned int) * m_iNrFaces * 3;
			bd.BindFlags = D3D11_BIND_INDEX_BUFFER;
			V( pd3dDev->CreateBuffer(&bd, &InitData, &m_pIndexStream) );

		}

		// 
		if(bGenQuadTree)
		{
			m_pQuadTree = new CQuadTree;
			if(m_pQuadTree!=NULL && m_pQuadTree->InitTree(m_iNrFaces))
			{
				for(int t=0; t<m_iNrFaces; t++)
					m_pQuadTree->AddTriangle(m_vVerts[m_iIndices[t*3+0]].vert, m_vVerts[m_iIndices[t*3+1]].vert, m_vVerts[m_iIndices[t*3+2]].vert);
				bool bRes = m_pQuadTree->BuildTree();
			}
		}


		if(m_vVerts!=NULL) { delete [] m_vVerts; m_vVerts=NULL; }
		if(m_iIndices!=NULL) { delete [] m_iIndices; m_iIndices=NULL; }
	}

	return bSuccess;
}

float CMeshFil::QueryTopY(const float fX, const float fZ) const
{
	float fRes = -10000000000.0f;
	if(m_pQuadTree!=NULL)
		fRes = m_pQuadTree->QueryTopY(fX, fZ);

	return fRes;
}

void CMeshFil::CleanUp()
{
	if(m_pVertStream!=NULL) SAFE_RELEASE( m_pVertStream );
	if(m_pIndexStream!=NULL) SAFE_RELEASE( m_pIndexStream );
}




CMeshFil::CMeshFil()
{
	m_iNrVerts = 0;
	m_iNrFaces = 0;
	m_vVerts = NULL;
	m_iIndices = NULL;

	m_pVertStream = NULL;
	m_pIndexStream = NULL;

	m_vMin = Vec3(0,0,0);
	m_vMax = Vec3(0,0,0);
}


CMeshFil::~CMeshFil()
{


}