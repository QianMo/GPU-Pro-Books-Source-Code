#ifndef __FLUID_RENDER_DESCRIPTION__H__
#define __FLUID_RENDER_DESCRIPTION__H__

#include "../Util/Color.h"

class TiXmlElement;


// -----------------------------------------------------------------------------
/// 
// -----------------------------------------------------------------------------
/// \ingroup 
/// 
/// 
// -----------------------------------------------------------------------------
class FluidRenderDescription
{
public:
	FluidRenderDescription(void);
	~FluidRenderDescription(void);

	void Reset();

	void LoadFromFile(const char* fileName);
	void SaveToFile(const char* fileName) const;

public:

	bool renderAABB;

	/// Particle size
	float particleSize;

	/// Base color of the fluid
	Color baseColor;
	
	/// Color falloff (depending on thickness)
	Color colorFalloff;
	/// Amount by which the color falloff is scale during final rendering of the fluid
	float falloffScale;

	/// Specular highlights
	Color specularColor;
	float specularShininess;

	/// Color of the spray particles
	Color sprayColor;
	/// Particles below that density threshold are rendered as spray
	float densityThreshold;

	/// Fresnel parameters
	float fresnelBias;
	float fresnelScale;
	float fresnelPower;

	/// Scale that is applied to the thickness before the refraction is calculated
	float thicknessRefraction;

	/// Scales that are applied during the thickness passes
	float fluidThicknessScale;
	float foamThicknessScale;

	/// Curvature filtering parameters
	float worldSpaceKernelRadius;

	/// Flags if noise is used during rendering
	bool useNoise;

	/// Noise control parameters
	float noiseDepthFalloff;
	float normalNoiseWeight;

	/// Foam base colors
	Color foamBackColor;
	Color foamFrontColor;

	/// Overall opacity/scale of the foam
	float foamFalloffScale;
	/// Weber number threshold
	float foamThreshold;
	/// Lifetime of a foam particle (applied when a particle changes from fluid to foam)
	float foamLifetime;
	/// Parameter indicating the thickness of the foam front layer
	float foamDepthThreshold;
	/// Opacity/scale of the foam front layer
	float foamFrontFalloffScale;
};

#endif

