#include "DXUT.h"
#include "ShadingMaterial.h"
#include "Rendition.h"
#include "RenderContext.h"
#include "Theatre.h"

ShadingMaterial::ShadingMaterial(Rendition* rendition, ID3D10InputLayout* inputLayout)
{
	this->rendition = rendition;
	this->inputLayout = inputLayout;

	// chnage this to use inputlayoutmanager
/*	D3D10_PASS_DESC passDesc;
	rendition->getTechnique()->GetPassByIndex(0)->GetDesc(&passDesc);
	const D3D10_INPUT_ELEMENT_DESC* elements;
	unsigned int nElements;
	mesh->GetVertexDescription(&elements, &nElements);
	device->CreateInputLayout(elements, nElements, passDesc.pIAInputSignature, passDesc.IAInputSignatureSize, &inputLayout);*/
}

ShadingMaterial::~ShadingMaterial(void)
{
}

void ShadingMaterial::apply(ID3D10Device* device, const EffectSettings& overrides)
{
	ID3D10EffectTechnique* technique = rendition->apply(device, overrides);
	technique->GetPassByIndex(0)->Apply(0);
	device->IASetInputLayout(inputLayout);
}

void ShadingMaterial::renderMesh(const RenderContext& context, ID3DX10Mesh* mesh, const EffectSettings& overrides)
{
	apply(context.theatre->getDevice(), overrides);

	if(context.instanceCount < 2)
		mesh->DrawSubset(0);
	else
		mesh->DrawSubsetInstanced(0, context.instanceCount, 0);

/*	D3D10_TECHNIQUE_DESC techniqueDesc;
	technique->GetDesc( &techniqueDesc );
    for(unsigned int p=0; p < techniqueDesc.Passes; p++)
    {
		technique->GetPassByIndex(p)->Apply(0);
		if(context.instanceCount < 2)
			mesh->DrawSubset(0);
		else
			mesh->DrawSubsetInstanced(0, context.instanceCount, 0);
    }*/
}