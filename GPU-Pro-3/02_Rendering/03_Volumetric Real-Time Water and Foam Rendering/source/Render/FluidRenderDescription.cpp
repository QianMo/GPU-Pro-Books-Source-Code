
#include "../Render/FluidRenderDescription.h"

#include "../XMLParser/tinyxml.h"
#include "../XMLParser/tinyutil.h"


// -----------------------------------------------------------------------------
// --------------- FluidRenderDescription::FluidRenderDescription --------------
// -----------------------------------------------------------------------------
FluidRenderDescription::FluidRenderDescription(void)
{
	Reset();
}

// -----------------------------------------------------------------------------
// -------------- FluidRenderDescription::~FluidRenderDescription --------------
// -----------------------------------------------------------------------------
FluidRenderDescription::~FluidRenderDescription(void)
{
}

// -----------------------------------------------------------------------------
// ----------------------- FluidRenderDescription::Reset -----------------------
// -----------------------------------------------------------------------------
void FluidRenderDescription::Reset(void)
{
	renderAABB = false;

	particleSize = 0.9375f;

	baseColor[0] = 0.305882f;
	baseColor[1] = 0.545098f;
	baseColor[2] = 0.956863f;
	baseColor[3] = 1.0f;

	colorFalloff[0] = 0.784314f;
	colorFalloff[1] = 0.784314f;
	colorFalloff[2] = 0.588235f;
	colorFalloff[3] = 1.0f;

	falloffScale = 0.02f;

	specularColor[0] = 1.0f;
	specularColor[1] = 1.0f;
	specularColor[2] = 1.0f;

	specularShininess = 80.0f;

	sprayColor[0] = 1.0f;
	sprayColor[1] = 1.0f;
	sprayColor[2] = 1.0f;
	sprayColor[3] = 0.4f;

	densityThreshold = 200.0f;

	fresnelBias = 0.1f;
	fresnelScale = 0.4f;
	fresnelPower = 2.0f;

	thicknessRefraction = 0.2f;

	fluidThicknessScale = 0.8f;
	foamThicknessScale = 0.8f;

	worldSpaceKernelRadius = 4.5f;


	useNoise = false;

	noiseDepthFalloff = 2.0f;
	normalNoiseWeight = 0.336f;

	// foam parameters
	foamBackColor[0] = 0.25f;
	foamBackColor[1] = 0.25f;
	foamBackColor[2] = 0.25f;
	foamBackColor[3] = 1.0f;

	foamFrontColor[0] = 1.0f;
	foamFrontColor[1] = 1.0f;
	foamFrontColor[2] = 1.0f;
	foamFrontColor[3] = 1.0f;

	foamFalloffScale = 0.5f;
	foamThreshold = 80.0f;
	foamLifetime = 1.0f;
	foamDepthThreshold = 10.0f;
	foamFrontFalloffScale = 0.333333f;
}

// -----------------------------------------------------------------------------
// -------------------- FluidRenderDescription::LoadFromFile -------------------
// -----------------------------------------------------------------------------
void FluidRenderDescription::LoadFromFile(const char* fileName)
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

		particleSize = TinyUtil::GetElement<float>(root.FirstChild("ParticleSize").Element());

		baseColor = TinyUtil::GetElement<Color>(root.FirstChild("BaseColor").Element());

		colorFalloff = TinyUtil::GetElement<Color>(root.FirstChild("ColorFalloff").Element());
		falloffScale = TinyUtil::GetElement<float>(root.FirstChild("FalloffScale").Element());

		specularColor = TinyUtil::GetElement<Color>(root.FirstChild("SpecularColor").Element());
		specularShininess = TinyUtil::GetElement<float>(root.FirstChild("SpecularShininess").Element());

		sprayColor = TinyUtil::GetElement<Color>(root.FirstChild("SprayColor").Element());
		densityThreshold = TinyUtil::GetElement<float>(root.FirstChild("DensityThreshold").Element());

		fresnelBias = TinyUtil::GetElement<float>(root.FirstChild("FresnelBias").Element());
		fresnelScale = TinyUtil::GetElement<float>(root.FirstChild("FresnelScale").Element());
		fresnelPower = TinyUtil::GetElement<float>(root.FirstChild("FresnelPower").Element());

		thicknessRefraction= TinyUtil::GetElement<float>(root.FirstChild("ThicknessRefraction").Element());

		fluidThicknessScale = TinyUtil::GetElement<float>(root.FirstChild("FluidThicknessScale").Element());
		foamThicknessScale = TinyUtil::GetElement<float>(root.FirstChild("FoamThicknessScale").Element());

		worldSpaceKernelRadius = TinyUtil::GetElement<float>(root.FirstChild("WorldSpaceKernelRadius").Element());

		useNoise = TinyUtil::GetElement<bool>(root.FirstChild("UseNoise").Element());

		noiseDepthFalloff = TinyUtil::GetElement<float>(root.FirstChild("NoiseDepthFalloff").Element());
		normalNoiseWeight = TinyUtil::GetElement<float>(root.FirstChild("NormalNoiseWeight").Element());

		foamBackColor = TinyUtil::GetElement<Color>(root.FirstChild("FoamBackColor").Element());
		foamFrontColor = TinyUtil::GetElement<Color>(root.FirstChild("FoamFrontColor").Element());

		foamFalloffScale = TinyUtil::GetElement<float>(root.FirstChild("FoamFalloffScale").Element());
		foamThreshold = TinyUtil::GetElement<float>(root.FirstChild("FoamThreshold").Element());
		foamLifetime = TinyUtil::GetElement<float>(root.FirstChild("FoamLifetime").Element());
		foamDepthThreshold = TinyUtil::GetElement<float>(root.FirstChild("FoamDepthThreshold").Element());
		foamFrontFalloffScale = TinyUtil::GetElement<float>(root.FirstChild("FoamFrontFalloffScale").Element());
	}
}

// -----------------------------------------------------------------------------
// --------------------- FluidRenderDescription::SaveToFile --------------------
// -----------------------------------------------------------------------------
void FluidRenderDescription::SaveToFile(const char* fileName) const
{
	TiXmlDocument doc;
	TiXmlDeclaration * decl = new TiXmlDeclaration( "1.0", "", "" );
	doc.LinkEndChild( decl );

	TiXmlElement* element = new TiXmlElement( "RenderDescription" );
	doc.LinkEndChild( element );

	TinyUtil::AddElement<float>(element, "ParticleSize", particleSize);

	TinyUtil::AddElement<Color>(element, "BaseColor", baseColor);

	TinyUtil::AddElement<Color>(element, "ColorFalloff", colorFalloff);
	TinyUtil::AddElement<float>(element, "FalloffScale", falloffScale);

	TinyUtil::AddElement<Color>(element, "SpecularColor", specularColor);
	TinyUtil::AddElement<float>(element, "SpecularShininess", specularShininess);

	TinyUtil::AddElement<Color>(element, "SprayColor", sprayColor);
	TinyUtil::AddElement<float>(element, "DensityThreshold", densityThreshold);

	TinyUtil::AddElement<float>(element, "FresnelBias", fresnelBias);
	TinyUtil::AddElement<float>(element, "FresnelScale", fresnelScale);
	TinyUtil::AddElement<float>(element, "FresnelPower", fresnelPower);

	TinyUtil::AddElement<float>(element, "ThicknessRefraction", thicknessRefraction);

	TinyUtil::AddElement<float>(element, "FluidThicknessScale", fluidThicknessScale);
	TinyUtil::AddElement<float>(element, "FoamThicknessScale", foamThicknessScale);

	TinyUtil::AddElement<float>(element, "WorldSpaceKernelRadius", worldSpaceKernelRadius);

	TinyUtil::AddElement<bool>(element, "UseNoise", useNoise);

	TinyUtil::AddElement<float>(element, "NoiseDepthFalloff", noiseDepthFalloff);
	TinyUtil::AddElement<float>(element, "NormalNoiseWeight", normalNoiseWeight);

	TinyUtil::AddElement<Color>(element, "FoamBackColor", foamBackColor);
	TinyUtil::AddElement<Color>(element, "FoamFrontColor", foamFrontColor);

	TinyUtil::AddElement<float>(element, "FoamFalloffScale", foamFalloffScale);
	TinyUtil::AddElement<float>(element, "FoamThreshold", foamThreshold);
	TinyUtil::AddElement<float>(element, "FoamLifetime", foamLifetime);
	TinyUtil::AddElement<float>(element, "FoamDepthThreshold", foamDepthThreshold);
	TinyUtil::AddElement<float>(element, "FoamFrontFalloffScale", foamFrontFalloffScale);

	doc.SaveFile(fileName);
}
