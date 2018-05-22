#ifndef __FLUID_DESCRIPTION__H__
#define __FLUID_DESCRIPTION__H__

#include "../Util/Color.h"

class TiXmlElement;

class Fluid;


// -----------------------------------------------------------------------------
/// 
// -----------------------------------------------------------------------------
/// \ingroup 
/// 
/// 
// -----------------------------------------------------------------------------
class FluidDescription
{
public:
	FluidDescription(void);
	~FluidDescription(void);

	/// Sets default values
	void Reset();

	/// Load parameters from file
	void LoadFromFile(const char* fileName);
	/// Save parameters to file
	void SaveToFile(const char* fileName) const;

	/// Load parameters from fluid class
	void LoadFromFluid(Fluid* fluid);

	/// Check if parameters are valid
	bool IsValid(void) const;

public:

	unsigned int	maxParticles;
	unsigned int	numReserveParticles;
	float			restParticlesPerMeter;
	float			restDensity;
	float			kernelRadiusMultiplier;
	float			motionLimitMultiplier;
	float			collisionDistanceMultiplier;

	//unsigned int	packetSizeMultiplier;

	float			stiffness;
	float			viscosity;
	float			surfaceTension;
	float			damping;
	float			fadeInTime;

	//Vector3		externalAcceleration;
	//NxPlane		projectionPlane;

	float			restitutionForStaticShapes;
	float			dynamicFrictionForStaticShapes;

	//float			staticFrictionForStaticShapes;
	//float			attractionForStaticShapes;

	float			restitutionForDynamicShapes;
	float			dynamicFrictionForDynamicShapes;

	//float			staticFrictionForDynamicShapes;
	//float			attractionForDynamicShapes;

	float			collisionResponseCoefficient;
	unsigned int	simulationMethod;
};

#endif

