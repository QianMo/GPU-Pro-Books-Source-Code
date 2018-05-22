
#include "../Physic/FluidDescription.h"
#include "../Physic/Fluid.h"

#include "NxPhysics.h"

#include "../XMLParser/tinyxml.h"
#include "../XMLParser/tinyutil.h"


// -----------------------------------------------------------------------------
// --------------------- FluidDescription::FluidDescription --------------------
// -----------------------------------------------------------------------------
FluidDescription::FluidDescription(void)
{
	Reset();
}

// -----------------------------------------------------------------------------
// -------------------- FluidDescription::~FluidDescription --------------------
// -----------------------------------------------------------------------------
FluidDescription::~FluidDescription(void)
{
}

// -----------------------------------------------------------------------------
// -------------------------- FluidDescription::Reset --------------------------
// -----------------------------------------------------------------------------
void FluidDescription::Reset(void)
{
	maxParticles						= 65535;
	numReserveParticles					= 0;
	restParticlesPerMeter				= 0.7f;
	restDensity							= 1000.0f;
	kernelRadiusMultiplier				= 2.0f;
	motionLimitMultiplier				= 16.0f;
	collisionDistanceMultiplier			= 0.75f;

	stiffness							= 10.0f;
	viscosity							= 15.0f;
	surfaceTension						= 0.0f;
	damping								= 0.0f;
	fadeInTime							= 0.0f;

	restitutionForStaticShapes			= 0.5f;
	dynamicFrictionForStaticShapes		= 0.0f;

	restitutionForDynamicShapes			= 0.0f;
	dynamicFrictionForDynamicShapes		= 0.0f;

	collisionResponseCoefficient		= 0.5f;
	simulationMethod					= NX_F_SPH;
}

// -----------------------------------------------------------------------------
// ----------------------- FluidDescription::LoadFromFile ----------------------
// -----------------------------------------------------------------------------
void FluidDescription::LoadFromFile(const char* fileName)
{
	TiXmlDocument document(fileName);
	if (!document.LoadFile())
	{
		Reset();
		return;
	}

	TiXmlHandle doc(&document);
	TiXmlElement* element;
	TiXmlHandle root(NULL);

	{
		element = doc.FirstChildElement().Element();
		if (!element)
		{
			assert(false);
			return;
		}
		root = TiXmlHandle(element);

		maxParticles = TinyUtil::GetElement<unsigned int>(root.FirstChild("MaxParticles").Element());
		numReserveParticles = TinyUtil::GetElement<unsigned int>(root.FirstChild("NumReserveParticles").Element());
		restParticlesPerMeter = TinyUtil::GetElement<float>(root.FirstChild("RestParticlesPerMeter").Element());
		restDensity = TinyUtil::GetElement<float>(root.FirstChild("RestDensity").Element());
		kernelRadiusMultiplier = TinyUtil::GetElement<float>(root.FirstChild("KernelRadiusMultiplier").Element());
		motionLimitMultiplier = TinyUtil::GetElement<float>(root.FirstChild("MotionLimitMultiplier").Element());
		collisionDistanceMultiplier = TinyUtil::GetElement<float>(root.FirstChild("CollisionDistanceMultiplier").Element());

		stiffness = TinyUtil::GetElement<float>(root.FirstChild("Stiffness").Element());
		viscosity = TinyUtil::GetElement<float>(root.FirstChild("Viscosity").Element());
		surfaceTension = TinyUtil::GetElement<float>(root.FirstChild("SurfaceTension").Element());
		damping = TinyUtil::GetElement<float>(root.FirstChild("Damping").Element());
		fadeInTime = TinyUtil::GetElement<float>(root.FirstChild("FadeInTime").Element());

		restitutionForStaticShapes = TinyUtil::GetElement<float>(root.FirstChild("RestitutionForStaticShapes").Element());
		dynamicFrictionForStaticShapes = TinyUtil::GetElement<float>(root.FirstChild("DynamicFrictionForStaticShapes").Element());

		restitutionForDynamicShapes = TinyUtil::GetElement<float>(root.FirstChild("RestitutionForDynamicShapes").Element());
		dynamicFrictionForDynamicShapes = TinyUtil::GetElement<float>(root.FirstChild("DynamicFrictionForDynamicShapes").Element());

		collisionResponseCoefficient = TinyUtil::GetElement<float>(root.FirstChild("CollisionResponseCoefficient").Element());
		simulationMethod = TinyUtil::GetElement<unsigned int>(root.FirstChild("SimulationMethod").Element());
	}
}

// -----------------------------------------------------------------------------
// ------------------------ FluidDescription::SaveToFile -----------------------
// -----------------------------------------------------------------------------
void FluidDescription::SaveToFile(const char* fileName) const
{
	TiXmlDocument doc;
	TiXmlDeclaration * decl = new TiXmlDeclaration( "1.0", "", "" );
	doc.LinkEndChild( decl );

	TiXmlElement* element = new TiXmlElement( "RenderDescription" );
	doc.LinkEndChild( element );

	TinyUtil::AddElement<unsigned int>(element, "MaxParticles", maxParticles);
	TinyUtil::AddElement<unsigned int>(element, "NumReserveParticles", numReserveParticles);
	TinyUtil::AddElement<float>(element, "RestParticlesPerMeter", restParticlesPerMeter);
	TinyUtil::AddElement<float>(element, "RestDensity", restDensity);
	TinyUtil::AddElement<float>(element, "KernelRadiusMultiplier", kernelRadiusMultiplier);
	TinyUtil::AddElement<float>(element, "MotionLimitMultiplier", motionLimitMultiplier);
	TinyUtil::AddElement<float>(element, "CollisionDistanceMultiplier", collisionDistanceMultiplier);

	TinyUtil::AddElement<float>(element, "Stiffness", stiffness);
	TinyUtil::AddElement<float>(element, "Viscosity", viscosity);
	TinyUtil::AddElement<float>(element, "SurfaceTension", surfaceTension);
	TinyUtil::AddElement<float>(element, "Damping", damping);
	TinyUtil::AddElement<float>(element, "FadeInTime", fadeInTime);

	TinyUtil::AddElement<float>(element, "RestitutionForStaticShapes", restitutionForStaticShapes);
	TinyUtil::AddElement<float>(element, "DynamicFrictionForStaticShapes", dynamicFrictionForStaticShapes);

	TinyUtil::AddElement<float>(element, "RestitutionForDynamicShapes", restitutionForDynamicShapes);
	TinyUtil::AddElement<float>(element, "DynamicFrictionForDynamicShapes", dynamicFrictionForDynamicShapes);

	TinyUtil::AddElement<float>(element, "CollisionResponseCoefficient", collisionResponseCoefficient);
	TinyUtil::AddElement<unsigned int>(element, "SimulationMethod", simulationMethod);

	doc.SaveFile(fileName);
}

// -----------------------------------------------------------------------------
// ---------------------- FluidDescription::LoadFromFluid ----------------------
// -----------------------------------------------------------------------------
void FluidDescription::LoadFromFluid(Fluid* fluid)
{
	NxFluidDesc tmp;
	fluid->GetNxFluid()->saveToDesc(tmp);

	maxParticles						= tmp.maxParticles;
	numReserveParticles					= tmp.numReserveParticles;
	restParticlesPerMeter				= tmp.restParticlesPerMeter;
	restDensity							= tmp.restDensity;
	kernelRadiusMultiplier				= tmp.kernelRadiusMultiplier;
	motionLimitMultiplier				= tmp.motionLimitMultiplier;
	collisionDistanceMultiplier			= tmp.collisionDistanceMultiplier;
	//packetSizeMultiplier				= tmp.packetSizeMultiplier;
	stiffness							= tmp.stiffness;
	viscosity							= tmp.viscosity;
	surfaceTension						= tmp.surfaceTension;
	damping								= tmp.damping;
	fadeInTime							= tmp.fadeInTime;
	restitutionForStaticShapes			= tmp.restitutionForStaticShapes;
	dynamicFrictionForStaticShapes		= tmp.dynamicFrictionForStaticShapes;
	restitutionForDynamicShapes			= tmp.restitutionForDynamicShapes;
	dynamicFrictionForDynamicShapes		= tmp.dynamicFrictionForDynamicShapes;
	collisionResponseCoefficient		= tmp.collisionResponseCoefficient;
	simulationMethod					= tmp.simulationMethod;
}

// -----------------------------------------------------------------------------
// ------------------------- FluidDescription::IsValid -------------------------
// -----------------------------------------------------------------------------
bool FluidDescription::IsValid(void) const
{
	NxFluidDesc tmp;
	tmp.setToDefault();

	tmp.maxParticles					= maxParticles;
	tmp.numReserveParticles				= numReserveParticles;
	tmp.restParticlesPerMeter			= restParticlesPerMeter;
	tmp.restDensity						= restDensity;
	tmp.kernelRadiusMultiplier			= kernelRadiusMultiplier;
	tmp.motionLimitMultiplier			= motionLimitMultiplier;
	tmp.collisionDistanceMultiplier		= collisionDistanceMultiplier;
	//tmp.packetSizeMultiplier			= packetSizeMultiplier;
	tmp.stiffness						= stiffness;
	tmp.viscosity						= viscosity;
	tmp.surfaceTension					= surfaceTension;
	tmp.damping							= damping;
	tmp.fadeInTime						= fadeInTime;
	tmp.restitutionForStaticShapes		= restitutionForStaticShapes;
	tmp.dynamicFrictionForStaticShapes	= dynamicFrictionForStaticShapes;
	tmp.restitutionForDynamicShapes		= restitutionForDynamicShapes;
	tmp.dynamicFrictionForDynamicShapes	= dynamicFrictionForDynamicShapes;
	tmp.collisionResponseCoefficient	= collisionResponseCoefficient;
	tmp.simulationMethod				= simulationMethod;

	return tmp.isValid();
}
