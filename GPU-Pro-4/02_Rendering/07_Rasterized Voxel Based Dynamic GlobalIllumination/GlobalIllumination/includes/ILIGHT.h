#ifndef ILIGHT_H
#define ILIGHT_H

#include <GLOBAL_ILLUM.h>

enum lightTypes
{
	POINT_LT=0,
	DIRECTIONAL_LT
};

class SURFACE;
class DX11_UNIFORM_BUFFER;

// ILIGHT
//   Interface for different light types.
class ILIGHT
{
public:	
	ILIGHT()
	{
		index = 0;
		active = true;
		hasShadow = false;
		performUpdate = true; 
		globalIllumPP = NULL;
	}

	virtual ~ILIGHT()
	{
	}

	virtual lightTypes GetLightType() const=0;

	virtual void Update()=0;

	virtual void SetupShadowMapSurface(SURFACE *surface)=0;

	// adds surface for direct illumination
	virtual void AddLitSurface()=0;

	// adds surfaces for indirect illumination
	virtual void AddGridSurfaces()=0;

	virtual DX11_UNIFORM_BUFFER* GetUniformBuffer() const=0;
	
	int GetIndex() const
	{
		return index;
	}

	void SetActive(bool active) 
	{
		this->active = active;
	}

	bool IsActive() const
	{
		return active;
	}

	bool HasShadow() const
	{
		return hasShadow;
	}

protected:
	virtual void CalculateMatrices()=0;

	virtual void UpdateUniformBuffer()=0;

	int index;
	bool active; 
	bool hasShadow;
	bool performUpdate;
	GLOBAL_ILLUM *globalIllumPP;
	
};

#endif
