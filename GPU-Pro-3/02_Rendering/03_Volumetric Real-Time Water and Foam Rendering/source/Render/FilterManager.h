#ifndef __FILTER_MANAGER__H__
#define __FILTER_MANAGER__H__

#include "../Util/Singleton.h"

class FilterManager : public Singleton<FilterManager>
{
	friend class Singleton<FilterManager>;

public:
	FilterManager(void);

	/// Init shadow stuff
	void Init(unsigned int _mapSize=512, float _reconstructionOrder=16.0f, bool _useMipMaps=false);

	/// Exit
	void Exit(void);

	/// Generate mip maps
	void GenerateMipMaps(unsigned int texture);
	void GenerateMipMapsArray(unsigned int textureArray);

	/// Generate summed area table
	void GenerateSummedAreaTable(unsigned int texture);
	void GenerateSummedAreaTableArray(unsigned int textureArray);

	void SetStates(void) const;

	/// Sat sample count (the more samples the less passes)
	void SetSatSampleCount(unsigned int _count); // 0...3
	unsigned int GetSatSampleCount(void) const { return satSampleCount; }

private:

	void DrawQuad(void) const;
	void CheckFrameBufferState(void) const;

	bool isInizialized;

	unsigned int mapSize;
	unsigned int reconstructionOrder;
	bool useMipMaps;

	// 0 ... 2 samples
	// 1 ... 4 samples
	// 2 ... 8 samples
	// 3 ... 16 samples
	unsigned int satSampleCount; // for array lookup
	unsigned int satPassCount; // 2.0f*Log(mapSize)/Log(satSampleCountPass)

	unsigned int satBuffer;
	unsigned int satTexture;
};

#endif