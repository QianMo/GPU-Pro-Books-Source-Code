#ifndef __LIGHTTILING_H__
#define __LIGHTTILING_H__

struct SFiniteLightData;
struct SFiniteLightBound;

#include <geommath/geommath.h>


class CLightTiler
{
public:
	void InitFrame(const Mat44 &mWorldToCam, const Mat44 &mProjection);
	void InitTiler();
	void CompileLightList();
	void AddLight(const SFiniteLightData &lightData, const SFiniteLightBound &coliData);
	const SFiniteLightBound * GetOrderedBoundsList() const { return m_pLightColiDataOrdered; }
	const SFiniteLightData * GetLightsDataList() const { return m_pLightDataOrdered; }
	const Vec3 * GetScrBoundsList() const { return m_pScrBounds; }


	CLightTiler();
	~CLightTiler();

private:
	int m_iIndex, m_iNrVisibLights;

	SFiniteLightData * m_pLightDataScattered;
	SFiniteLightData * m_pLightDataOrdered;

	SFiniteLightBound * m_pLightColiDataScattered;
	SFiniteLightBound * m_pLightColiDataOrdered;

	Vec3 * m_pScrBounds;

	Mat44 m_mWorldToCam;
	Mat44 m_mProjection;
};


#endif