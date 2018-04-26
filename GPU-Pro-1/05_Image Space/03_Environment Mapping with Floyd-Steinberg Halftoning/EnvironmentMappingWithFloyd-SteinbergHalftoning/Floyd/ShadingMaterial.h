#pragma once

class Rendition;
class RenderContext;
class EffectSettings;

class ShadingMaterial
{
	Rendition* rendition;
	ID3D10InputLayout* inputLayout;

public:
	ShadingMaterial(Rendition* rendition, ID3D10InputLayout* inputLayout);
	~ShadingMaterial(void);

	void apply(ID3D10Device* device, const EffectSettings& overrides);

	void renderMesh(const RenderContext& context, ID3DX10Mesh* mesh, const EffectSettings& overrides);
};
