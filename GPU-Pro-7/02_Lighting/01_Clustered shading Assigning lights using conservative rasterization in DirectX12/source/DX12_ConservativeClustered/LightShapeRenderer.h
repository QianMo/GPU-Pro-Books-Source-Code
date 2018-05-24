#pragma once
#include "KGraphicsDevice.h"

class LightShapeRenderer
{
public:
	LightShapeRenderer();
	~LightShapeRenderer();

	ID3D12PipelineState* GetPSO()		{ return m_PSO; }
	ID3D12PipelineState* GetSpotPSO()	{ return m_SpotPSO; }
	ID3D12RootSignature* GetRootSig()	{ return m_RootSignature; }

private:
	ID3D12PipelineState* m_PSO;
	ID3D12PipelineState* m_SpotPSO;
	ID3D12RootSignature* m_RootSignature;
};