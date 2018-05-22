#ifndef __SHADOW_MANAGER__H__
#define __SHADOW_MANAGER__H__

#include <vector>

#include "../Util/Singleton.h"
#include "../Util/Matrix4.h"

#include "../Level/Light.h"

class Camera;
class ShadowMap;

class ShadowManager : public Singleton<ShadowManager>
{
	friend class Singleton<ShadowManager>;

public:

	ShadowManager(void);

	/// Init shadow stuff
	void Init(unsigned int _mapSize=512, float _reconstructionOrder=16.0f);

	/// Update
	void Update(float deltaTime);
	void UpdateSizeOfLight(float sizeOfLight);

	/// Exit
	void Exit(void);

	/// Render shadow map
	void BeginShadow(void);
	void ShadowPass(unsigned int passCount);
	void EndShadow(void);

	/// Prepare stuff for final pass
	void PrepareFinalPass(Matrix4& viewMatrix, unsigned int passCount);
	Light* GetCurrentLight(unsigned int passCount) const { return lights[passCount]; }

	/// Returns the pass count for the different render passes
	int GetPassCount(void) const { return (int)shadowMaps.size(); }

	/// Compute the convolution map from the shadow map
	void ComputeConvolutionMap(void);

private:

	bool isInizialized;

	unsigned int mapSize;
	float reconstructionOrder;

	std::vector<Light*> lights;
	std::vector<ShadowMap*> shadowMaps;
};

#endif