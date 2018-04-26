#pragma once
#include "EffectSettings.h"

class RenderContext;

/// Rendition class containing technique and effect variable settings for a submesh.
class Rendition : public EffectSettings
{
	/// The name identifying the technique in the effect file.
	ID3D10EffectTechnique* technique;

	/// Null if no base.
	Rendition* base;

public:
	/// Constructor.
	Rendition(ID3D10Device* device, ID3D10EffectTechnique* technique);
	Rendition(Rendition* base);
	~Rendition();

	/// Sets the technique name and effect variables according to the stored values, using the effect interface.
	ID3D10EffectTechnique* apply(ID3D10Device* device, const EffectSettings& overrides);

	ID3D10EffectTechnique* getTechnique()
	{
		if(technique)
			return technique;
		else
			return base->getTechnique();
	}

};
