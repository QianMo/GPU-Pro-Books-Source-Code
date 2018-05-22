#pragma once

#include "Core.h"
#include "Effects.h"

struct DeepShadowMapEffect
{
	ID3DX11Effect								*dsmEffect;
	ID3DX11EffectTechnique						*dsmRenderTechnique, *dsmSortTechnique, *dsmLinkTechnique;
	ID3DX11EffectMatrixVariable					*dsmEffectWorldViewProj;
	ID3DX11EffectUnorderedAccessViewVariable	*dsmEffectLinkedListBufWP, *dsmEffectLinkedListBufDAN, *dsmEffectNeighborsBuf, *dsmEffectStartElementBuf;
	ID3DX11EffectScalarVariable					*dsmEffectDimension, *dsmEffectAlpha;
	ID3DX11EffectShaderResourceVariable			*dsmEffectLinkedListBufWPRO, *dsmEffectLinkedListBufDANRO, *dsmEffectNeighborsBufRO, *dsmEffectStartElementBufRO;
};

class DeepShadowMap : public ICoreBase
{
public:
	DeepShadowMap();

	CoreResult Init(Core *core, UINT dim, DWORD numElementsInBuffer);

	void Set(CoreCamera &lightCam, CoreMatrix4x4 &world, float alpha, int currentEffect);

	void SetShaderForRealRendering(ID3DX11EffectScalarVariable *effectDimension, ID3DX11EffectShaderResourceVariable *effectLinkedListBufWPRO, ID3DX11EffectShaderResourceVariable *effectNeighborsBufRO, ID3DX11EffectShaderResourceVariable *effectStartElementBuf);
	void UnsetShaderForRealRendering(ID3DX11EffectScalarVariable *effectDimension, ID3DX11EffectShaderResourceVariable *effectLinkedListBufWPRO, ID3DX11EffectShaderResourceVariable *effectNeighborsBufRO, ID3DX11EffectShaderResourceVariable *effectStartElementBuf);

	void SortLists(int currentEffect);
	void LinkLists(int currentEffect);

	void ChangeAlpha(float alpha, int currentEffect);
	void ChangeLightCamera(CoreCamera &lightCam, CoreMatrix4x4 &world, int currentEffect);

	void Unset(int currentEffect);

protected:
	D3D11_VIEWPORT				vp;
	Core						*core;
	ID3D11Buffer				*linkedListBuf, *startElementBuf;
	ID3D11Buffer				*linkedListBuf2, *linkedListBuf3;
	ID3D11ShaderResourceView	*linkedListBufSRV, *startElementBufSRV;
	ID3D11UnorderedAccessView	*linkedListBufUAV, *startElementBufUAV;
	ID3D11ShaderResourceView	*linkedListBuf2SRV, *linkedListBuf3SRV;
	ID3D11UnorderedAccessView	*linkedListBuf2UAV, *linkedListBuf3UAV;

	DeepShadowMapEffect			dsmEffects[NUM_EFFECTS];
	

	// Release everything
	void finalRelease();

};