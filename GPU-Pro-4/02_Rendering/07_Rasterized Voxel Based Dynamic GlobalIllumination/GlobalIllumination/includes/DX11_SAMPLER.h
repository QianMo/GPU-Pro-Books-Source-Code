#ifndef DX11_SAMPLER_H
#define DX11_SAMPLER_H

#include <render_states.h>

// descriptor for setting up DX11_SAMPLER
struct SAMPLER_DESC
{
	SAMPLER_DESC()
	{
		filter = MIN_MAG_LINEAR_FILTER;
		maxAnisotropy = 2;
		adressU = CLAMP_TEX_ADRESS;
		adressV = CLAMP_TEX_ADRESS;
		adressW = CLAMP_TEX_ADRESS;
		borderColor.Set(0.0f,0.0f,0.0f,0.0f);
		minLOD = -FLT_MAX;
		maxLOD = FLT_MAX;
		lodBias = 0.0f;
		compareFunc = LEQUAL_COMP_FUNC;
	}

	bool operator== (const SAMPLER_DESC &desc) const
	{
		if(filter!=desc.filter)
			return false;
		if(maxAnisotropy!=desc.maxAnisotropy)
			return false;
		if(adressU!=desc.adressU)
			return false;
		if(adressV!=desc.adressV)
			return false;
		if(adressW!=desc.adressW)
			return false;
		if(borderColor!=desc.borderColor)
			return false;
		if(!IS_EQUAL(minLOD,desc.minLOD))
			return false;
		if(!IS_EQUAL(maxLOD,desc.maxLOD))
			return false;
		if(!IS_EQUAL(lodBias,desc.lodBias))
			return false;
		if(compareFunc!=desc.compareFunc)
			return false;
		return true;
	}

	filterModes filter;
	unsigned int maxAnisotropy;
	texAdressModes adressU;
	texAdressModes adressV;
	texAdressModes adressW;
	COLOR borderColor;
	float minLOD;
	float maxLOD;
	float lodBias;
	comparisonFuncs compareFunc;
};

// DX11_SAMPLER
//   Wrapper for ID3D11SamplerState.
class DX11_SAMPLER
{
public:
	DX11_SAMPLER()
	{
		sampler = NULL;
	}

	~DX11_SAMPLER()
	{
		Release();
	}

	void Release();

	bool Create(const SAMPLER_DESC &desc);

	void Bind(shaderTypes shaderType,textureBP bindingPoint) const;

	SAMPLER_DESC GetDesc() const
	{
		return desc;
	}

private:
	SAMPLER_DESC desc;
	ID3D11SamplerState *sampler;

};

#endif