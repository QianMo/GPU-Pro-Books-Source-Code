#include "ClusterRenderer.h"
#include "SharedContext.h"
#include "Log.h"
#include "d3dx12.h"

using namespace Log;

ClusterRenderer::ClusterRenderer()
	: m_numPoints(0)
{
	HRESULT hr;

	const uint32 max_num_line_points = 500000;

	m_lineBatch.resize(max_num_line_points);

	KShader vertexShader(L"../assets/shaders/ClusterLines.vertex", "main", "vs_5_0");
	KShader pixelShader(L"../assets/shaders/ClusterLines.pixel", "main", "ps_5_0");

	RootDescriptorRange root_desc_range[] =
	{
		{ 1, ROOT_DESCRIPTOR_TYPE::RANGE_CBV, SHADER_VISIBILITY::VERTEX },
		{ 1, ROOT_DESCRIPTOR_TYPE::RANGE_SRV, SHADER_VISIBILITY::ALL },
	};

	m_rootSig = shared_context.gfx_device->CreateRootSignature(2, root_desc_range);

	D3D12_RASTERIZER_DESC rasterDescFront;
	rasterDescFront.AntialiasedLineEnable = FALSE;
	rasterDescFront.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
	rasterDescFront.CullMode = D3D12_CULL_MODE_NONE;
	rasterDescFront.DepthBias = D3D12_DEFAULT_DEPTH_BIAS;
	rasterDescFront.DepthBiasClamp = D3D12_DEFAULT_DEPTH_BIAS_CLAMP;
	rasterDescFront.DepthClipEnable = TRUE;
	rasterDescFront.FillMode = D3D12_FILL_MODE_SOLID;
	rasterDescFront.ForcedSampleCount = 0;
	rasterDescFront.FrontCounterClockwise = FALSE;
	rasterDescFront.MultisampleEnable = FALSE;
	rasterDescFront.SlopeScaledDepthBias = D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS;

	D3D12_GRAPHICS_PIPELINE_STATE_DESC PSODesc;
	ZeroMemory(&PSODesc, sizeof(PSODesc));
	PSODesc.InputLayout = vertexShader.GetInputLayout();
	PSODesc.pRootSignature = m_rootSig;
	PSODesc.VS = { reinterpret_cast<BYTE*>(vertexShader.GetBufferPointer()), vertexShader.GetBufferSize() };
	PSODesc.PS = { reinterpret_cast<BYTE*>(pixelShader.GetBufferPointer()), pixelShader.GetBufferSize() };
	PSODesc.RasterizerState = rasterDescFront;
	PSODesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
	PSODesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
	PSODesc.DepthStencilState.DepthEnable = TRUE;
	PSODesc.DepthStencilState.StencilEnable = FALSE;
	PSODesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
	PSODesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
	PSODesc.SampleMask = UINT_MAX;
	PSODesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
	PSODesc.NumRenderTargets = 1;
	PSODesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
	PSODesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
	PSODesc.SampleDesc.Count = 1;

	hr = shared_context.gfx_device->GetDevice()->CreateGraphicsPipelineState(&PSODesc, IID_PPV_ARGS(&m_PSO));
	if (FAILED(hr))
		PRINT(LogLevel::FATAL_ERROR, "Failed to create m_PSO");

	m_lineBufferCPUHandle = shared_context.gfx_device->GetDescHeapCBV_SRV()->GetNewCPUHandle();
	m_lineBufferGPUHandle = shared_context.gfx_device->GetDescHeapCBV_SRV()->GetGPUHandleAtHead();


	CD3DX12_RESOURCE_DESC line_buff_desc(D3D12_RESOURCE_DIMENSION_BUFFER, 0, max_num_line_points * sizeof(Vector3), 1, 1, 1,
		DXGI_FORMAT_UNKNOWN, 1, 0, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE);

	shared_context.gfx_device->GetDevice()->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD, 0, 0),
		D3D12_HEAP_FLAG_NONE,
		&line_buff_desc,
		D3D12_RESOURCE_STATE_GENERIC_READ,
		nullptr,
		IID_PPV_ARGS(&m_lineBuffer));

	//Map
	m_lineBuffer->Map(0, nullptr, (void**)&m_lineMem);

	//Also set up SRV for it
	D3D12_SHADER_RESOURCE_VIEW_DESC line_buff_srv_view_desc;
	ZeroMemory(&line_buff_srv_view_desc, sizeof(line_buff_srv_view_desc));
	line_buff_srv_view_desc.Format = DXGI_FORMAT_R32G32B32_FLOAT;
	line_buff_srv_view_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	line_buff_srv_view_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	line_buff_srv_view_desc.Buffer.FirstElement = 0;
	line_buff_srv_view_desc.Buffer.NumElements = max_num_line_points;
	line_buff_srv_view_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
	line_buff_srv_view_desc.Buffer.StructureByteStride = 0;

	shared_context.gfx_device->GetDevice()->CreateShaderResourceView(m_lineBuffer, &line_buff_srv_view_desc, m_lineBufferCPUHandle);

	//Frustum stuff
	m_HalfHeight = (float)std::tan(DirectX::XM_PIDIV4 * 0.5f);
	m_HalfWidth = m_HalfHeight * ((float)Constants::WINDOW_WIDTH / (float)Constants::WINDOW_HEIGHT);

	const float min_z = Constants::NEAR_CLUST;
	const float max_z = (float)Constants::FARZ;

	m_ln2 = logf(2.0f);
	m_ln2_inv = 1.0f / logf(2.0f);
	m_log2_min = logf(min_z) * m_ln2_inv;
	m_log2_max = logf(max_z) * m_ln2_inv;
}

ClusterRenderer::~ClusterRenderer()
{
	m_rootSig->Release();
	m_PSO->Release();
	m_lineBuffer->Release();
}

void ClusterRenderer::BuildWorldSpacePositions(Camera* camera, bool exp_depth_dist)
{
	//"Clear" linebuffer
	m_numPoints = 0;

	float frust_depth = Constants::FARZ - Constants::NEARZ;

	float clust_depth = frust_depth / Constants::NR_Z_CLUSTS;

	for(int z = 0; z < Constants::NR_Z_PLANES; ++z)
	{
		float dep;
		if(exp_depth_dist)
			dep = (z == 0) ? Constants::NEARZ : expf((float(z - 1) * (m_log2_max - m_log2_min) / float(Constants::NR_Z_CLUSTS - 1) + m_log2_min) * m_ln2);
		else
			dep = (Constants::NEARZ + clust_depth * z);

		//Find near top left position and cluster
		Vector3 right = m_HalfWidth * dep * camera->right;
		Vector3 top = m_HalfHeight * dep * camera->up;
		Vector3 center = camera->position + dep * camera->facing;

		Vector3 tl = center - right + top;
		Vector3 tr = center + right + top;
		Vector3 bl = center - right - top;

		Vector3 LtoR = (tl - tr) / (float)Constants::NR_X_CLUSTS;
		Vector3 TtoB = (bl - tl) / (float)Constants::NR_Y_CLUSTS;

		for (int y = 0; y < Constants::NR_Y_PLANES; ++y)
		{
			for (int x = 0; x < Constants::NR_X_PLANES; ++x)
			{
				m_worldSpaceClusterPoints[x][y][z] = Vector3(tr + (float)x*LtoR + (float)y*TtoB);
			}
		}
	}
}

void ClusterRenderer::AddCluster(uint32 x, uint32 y, uint32 z)
{
	if (m_numPoints > 500000 - 25)
		return;

	m_lineBatch[m_numPoints++] = m_worldSpaceClusterPoints[x][y][z];
	m_lineBatch[m_numPoints++] = m_worldSpaceClusterPoints[x + 1][y][z];

	m_lineBatch[m_numPoints++] = m_worldSpaceClusterPoints[x][y][z];
	m_lineBatch[m_numPoints++] = m_worldSpaceClusterPoints[x][y + 1][z];

	m_lineBatch[m_numPoints++] = m_worldSpaceClusterPoints[x][y][z];
	m_lineBatch[m_numPoints++] = m_worldSpaceClusterPoints[x][y][z + 1];

	m_lineBatch[m_numPoints++] = m_worldSpaceClusterPoints[x][y + 1][z];
	m_lineBatch[m_numPoints++] = m_worldSpaceClusterPoints[x][y + 1][z + 1];

	m_lineBatch[m_numPoints++] = m_worldSpaceClusterPoints[x][y + 1][z];
	m_lineBatch[m_numPoints++] = m_worldSpaceClusterPoints[x + 1][y + 1][z];

	m_lineBatch[m_numPoints++] = m_worldSpaceClusterPoints[x + 1][y][z];
	m_lineBatch[m_numPoints++] = m_worldSpaceClusterPoints[x + 1][y + 1][z];

	m_lineBatch[m_numPoints++] = m_worldSpaceClusterPoints[x + 1][y][z];
	m_lineBatch[m_numPoints++] = m_worldSpaceClusterPoints[x + 1][y][z + 1];

	m_lineBatch[m_numPoints++] = m_worldSpaceClusterPoints[x + 1][y][z + 1];
	m_lineBatch[m_numPoints++] = m_worldSpaceClusterPoints[x + 1][y + 1][z + 1];

	m_lineBatch[m_numPoints++] = m_worldSpaceClusterPoints[x + 1][y + 1][z + 1];
	m_lineBatch[m_numPoints++] = m_worldSpaceClusterPoints[x + 1][y + 1][z];

	m_lineBatch[m_numPoints++] = m_worldSpaceClusterPoints[x + 1][y + 1][z + 1];
	m_lineBatch[m_numPoints++] = m_worldSpaceClusterPoints[x][y + 1][z + 1];

	m_lineBatch[m_numPoints++] = m_worldSpaceClusterPoints[x][y + 1][z + 1];
	m_lineBatch[m_numPoints++] = m_worldSpaceClusterPoints[x][y][z + 1];

	m_lineBatch[m_numPoints++] = m_worldSpaceClusterPoints[x][y][z + 1];
	m_lineBatch[m_numPoints++] = m_worldSpaceClusterPoints[x + 1][y][z + 1];
}

void ClusterRenderer::UploadClusters()
{
	memcpy(m_lineMem, &m_lineBatch[0], m_numPoints*sizeof(Vector3));
}

