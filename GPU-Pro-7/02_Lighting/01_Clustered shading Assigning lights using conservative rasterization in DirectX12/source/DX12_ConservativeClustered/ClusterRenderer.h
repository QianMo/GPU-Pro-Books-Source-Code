#pragma once
#include "KGraphicsDevice.h"
#include <SimpleMath.h>
#include "Camera.h"
#include <memory>
#include <vector>
#include "Constants.h"

using namespace DirectX::SimpleMath;

class ClusterRenderer
{
public:
	ClusterRenderer();
	~ClusterRenderer();

	void BuildWorldSpacePositions(Camera* camera, bool exp_depth_dist = true);
	void AddCluster(uint32 x, uint32 y, uint32 z);
	void UploadClusters();

	D3D12_CPU_DESCRIPTOR_HANDLE GetCPUHandle()	{ return m_lineBufferCPUHandle; }
	D3D12_GPU_DESCRIPTOR_HANDLE GetGPUHandle()	{ return m_lineBufferGPUHandle; }
	ID3D12RootSignature* GetRootSig()			{ return m_rootSig; }
	ID3D12PipelineState* GetPSO()				{ return m_PSO; }
	uint32 GetNumPoints()						{ return m_numPoints; }

private:

	ID3D12RootSignature* m_rootSig;
	ID3D12PipelineState* m_PSO;

	ID3D12Resource* m_lineBuffer;
	uint8* m_lineMem;

	D3D12_CPU_DESCRIPTOR_HANDLE m_lineBufferCPUHandle;
	D3D12_GPU_DESCRIPTOR_HANDLE m_lineBufferGPUHandle;

	std::vector<Vector3> m_lineBatch;

	float m_HalfHeight;
	float m_HalfWidth;
	float m_ln2;
	float m_ln2_inv;
	float m_log2_min;
	float m_log2_max;

	uint32 m_numPoints;

	Vector3 m_worldSpaceClusterPoints[Constants::NR_X_PLANES][Constants::NR_Y_PLANES][Constants::NR_Z_PLANES];

};