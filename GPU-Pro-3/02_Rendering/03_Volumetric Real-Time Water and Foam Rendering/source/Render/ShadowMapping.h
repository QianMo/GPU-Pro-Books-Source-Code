#ifndef __SHADOWMAPPING__H__
#define __SHADOWMAPPING__H__

#include "../Util/Singleton.h"

class Light;
class Camera;

class ShadowMapping : public Singleton<ShadowMapping>
{
	friend class Singleton<ShadowMapping>;

public:
	ShadowMapping(void);

	void Init(Light* light, Camera* camera, int width=1024, int height=1024);

	void Begin(void);
	void End(void);

	void RenderDebug(void);
private:
	unsigned int shadowMapTexture;
	unsigned int shadowMapDepthBuffer;

	int width;
	int height;
	Light* light;
	Camera* camera;
};

#endif