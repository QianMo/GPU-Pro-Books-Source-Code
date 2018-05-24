#include "LightTiling.h"
#include <geommath/geommath.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include "light_definitions.h"


void CLightTiler::AddLight(const SFiniteLightData &lightData, const SFiniteLightBound &coliData)
{
	Mat33 m33WrldToCam;
	for(int r=0; r<3; r++)
	{
		Vec4 row = GetRow(m_mWorldToCam, r);
		SetRow(&m33WrldToCam, r, Vec3(row.x, row.y, row.z));
	}

	m_pLightDataScattered[m_iIndex] = lightData;
	m_pLightColiDataScattered[m_iIndex] = coliData;

	m_pLightDataScattered[m_iIndex].vLpos = m_mWorldToCam*lightData.vLpos;
	m_pLightDataScattered[m_iIndex].vLdir = m33WrldToCam*lightData.vLdir;
	m_pLightDataScattered[m_iIndex].vBoxAxisX = m33WrldToCam*lightData.vBoxAxisX;
	m_pLightDataScattered[m_iIndex].vBoxAxisZ = m33WrldToCam*lightData.vBoxAxisZ;

	const Vec3 c0 = coliData.vBoxAxisX;
	const Vec3 c1 = coliData.vBoxAxisY;
	const Vec3 c2 = coliData.vBoxAxisZ;

	m_pLightColiDataScattered[m_iIndex].vBoxAxisX = m33WrldToCam*c0;
	m_pLightColiDataScattered[m_iIndex].vBoxAxisY = m33WrldToCam*c1;
	m_pLightColiDataScattered[m_iIndex].vBoxAxisZ = m33WrldToCam*c2;

	const Vec3 vCen = m_mWorldToCam*coliData.vCen;
	m_pLightColiDataScattered[m_iIndex].vCen = vCen;
	

	++m_iIndex;
}


void CLightTiler::InitTiler()
{
	m_pLightDataScattered = new SFiniteLightData[MAX_NR_LIGHTS_PER_CAMERA];
	m_pLightDataOrdered = new SFiniteLightData[MAX_NR_LIGHTS_PER_CAMERA];

	m_pLightColiDataScattered = new SFiniteLightBound[MAX_NR_LIGHTS_PER_CAMERA];
	m_pLightColiDataOrdered = new SFiniteLightBound[MAX_NR_LIGHTS_PER_CAMERA];

	m_pScrBounds = new Vec3[MAX_NR_LIGHTS_PER_CAMERA*2];
}

// sort by type in linear time
void CLightTiler::CompileLightList()
{
	m_iNrVisibLights = m_iIndex;

	int numOfType[MAX_TYPES], lgtOffs[MAX_TYPES], curIndex[MAX_TYPES];
	for(int i=0; i<MAX_TYPES; i++) { numOfType[i]=0; curIndex[i]=0; }

	// determine number of lights of each type
	for(int l=0; l<m_iNrVisibLights; l++) 
	{
		const int iTyp = m_pLightDataScattered[l].uLightType;
		assert(iTyp>=0 && iTyp<MAX_TYPES);
		++numOfType[iTyp];
	}

	// determine offset for each type
	lgtOffs[0]=0;
	for(int i=1; i<MAX_TYPES; i++) lgtOffs[i]=lgtOffs[i-1]+numOfType[i-1];
	
	// sort lights by type
	for(int l=0; l<m_iNrVisibLights; l++) 
	{
		const int iTyp = m_pLightDataScattered[l].uLightType;
		assert(iTyp>=0 && iTyp<MAX_TYPES);
		
		const int offs = curIndex[iTyp]+lgtOffs[iTyp];
		m_pLightDataOrdered[offs] = m_pLightDataScattered[l];
		m_pLightColiDataOrdered[offs] = m_pLightColiDataScattered[l];
		
		++curIndex[iTyp];
	}

	for(int i=0; i<MAX_TYPES; i++) assert(curIndex[i]==numOfType[i]);		// sanity check
}


void CLightTiler::InitFrame(const Mat44 &mWorldToCam, const Mat44 &mProjection)
{
	m_mWorldToCam = mWorldToCam;
	m_mProjection = mProjection;

	m_iNrVisibLights = 0;
	m_iIndex = 0;
}



CLightTiler::CLightTiler()
{
	m_pLightDataScattered = NULL;
	m_pLightDataOrdered = NULL;

	m_pLightColiDataScattered = NULL;
	m_pLightColiDataOrdered = NULL;

	m_pScrBounds = NULL;

	m_iIndex = 0;
	m_iNrVisibLights = 0;
}

CLightTiler::~CLightTiler()
{
	delete [] m_pLightDataScattered;
	delete [] m_pLightDataOrdered;

	delete [] m_pLightColiDataScattered;
	delete [] m_pLightColiDataOrdered;

	delete [] m_pScrBounds;
}