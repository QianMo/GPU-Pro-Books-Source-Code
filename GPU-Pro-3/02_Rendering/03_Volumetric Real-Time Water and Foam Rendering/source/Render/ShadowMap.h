#ifndef __SHADOW_MAP__H__
#define __SHADOW_MAP__H__

class ShadowMap
{
public:
	ShadowMap(unsigned int _mapSize=512, float _reconstructionOrder=16.0f, bool _useMipMaps=false);
	~ShadowMap(void);

	void SetFrameBuffer(void) const;

	/// Compute CSM
	void ComputeCsm(void);

	/// Prefilter textures with mip maps or sat
	void PreFilter(void);

	/// Return textures
	unsigned int GetShadowMap(void) const { return linearShadowMapTexture; }
	unsigned int GetSinTexture(void) const { return csmSinTextureArray; }
	unsigned int GetCosTexture(void) const { return csmCosTextureArray; }

private:

	void DrawQuad(void) const;
	void CheckFrameBufferState(void) const;
	
	unsigned int frameBuffer;
	unsigned int csmSinBuffer;
	unsigned int csmCosBuffer;

	unsigned int shadowMapTexture;
	unsigned int linearShadowMapTexture;

	unsigned int csmSinTextureArray;
	unsigned int csmCosTextureArray;

	unsigned int mapSize;
	float reconstructionOrder;
	bool useMipMaps;
};

#endif