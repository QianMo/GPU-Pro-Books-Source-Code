#include "DXUT.h"
#include "Rendition.h"
#include "RenderContext.h"
#include "Theatre.h"

Rendition::Rendition(ID3D10Device* device, ID3D10EffectTechnique* technique)
:technique(technique)
{
	base = NULL;
}

Rendition::Rendition(Rendition* base)
{
	this->base = base;
	this->technique = base->technique;
}

Rendition::~Rendition()
{
}

ID3D10EffectTechnique* Rendition::apply(ID3D10Device* device, const EffectSettings& overrides)
{
	assert(base || technique);
	if(base)
		base->apply(device, overrides);

	// set all resources
	{
		ShaderResourceSettings::iterator i = shaderResourceSettings.begin();
		while(i != shaderResourceSettings.end())
		{
			if(!overrides.applyShaderResourceSetting(i->first))
				i->first->SetResource(i->second);
			i++;
		}
	}
	// set all vectors
	{
		VectorSettings::iterator i = vectorSettings.begin();
		while(i != vectorSettings.end())
		{
			if(!overrides.applyVectorSetting(i->first))
				i->first->SetFloatVector((float*)&i->second);
			i++;
		}
	}

	return technique;
}
