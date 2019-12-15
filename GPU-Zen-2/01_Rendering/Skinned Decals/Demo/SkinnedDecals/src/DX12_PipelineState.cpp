#include <stdafx.h>
#include <Demo.h>
#include <DX12_PipelineState.h>

static const char *semanticNames[] = { "POSITION", "TEXCOORD", "NORMAL", "TANGENT", "COLOR" };

bool DX12_PipelineState::Create(const PipelineStateDesc &desc, const char *name)
{
  this->desc = desc;

  if(desc.pipelineStateType == GRAPHICS_PIPELINE_STATE)
  {
    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = desc.rootSignature->GetRootSignature();
    psoDesc.VS = desc.graphics.shader->GetByteCode(VERTEX_SHADER);
    psoDesc.HS = desc.graphics.shader->GetByteCode(HULL_SHADER);
    psoDesc.DS = desc.graphics.shader->GetByteCode(DOMAIN_SHADER);
    psoDesc.GS = desc.graphics.shader->GetByteCode(GEOMETRY_SHADER);
    psoDesc.PS = desc.graphics.shader->GetByteCode(PIXEL_SHADER);

    assert(desc.graphics.inputLayout.numElementDescs <= MAX_NUM_INPUT_ELEMENT_DESCS);
    psoDesc.InputLayout.NumElements = desc.graphics.inputLayout.numElementDescs;
    D3D12_INPUT_ELEMENT_DESC inputElementDescs[MAX_NUM_INPUT_ELEMENT_DESCS];
    for(UINT i=0; i<desc.graphics.inputLayout.numElementDescs; i++)
    {
      inputElementDescs[i].SemanticName = semanticNames[desc.graphics.inputLayout.elementDescs[i].inputElement];
      inputElementDescs[i].SemanticIndex = 0;
      inputElementDescs[i].Format = RenderFormat::GetDx12RenderFormat(desc.graphics.inputLayout.elementDescs[i].format).srvFormat;
      inputElementDescs[i].InputSlot = desc.graphics.inputLayout.elementDescs[i].vertexBufferSlot;
      inputElementDescs[i].AlignedByteOffset = desc.graphics.inputLayout.elementDescs[i].offset;
      inputElementDescs[i].InputSlotClass = (desc.graphics.inputLayout.elementDescs[i].instanceDataStepRate == 0) ? 
                                            D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA : D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA;
      inputElementDescs[i].InstanceDataStepRate = desc.graphics.inputLayout.elementDescs[i].instanceDataStepRate;
    }
    psoDesc.InputLayout.pInputElementDescs = (desc.graphics.inputLayout.numElementDescs > 0) ? inputElementDescs : nullptr;

    psoDesc.RasterizerState.FillMode = static_cast<D3D12_FILL_MODE>(desc.graphics.rasterizerDesc.fillMode);
    psoDesc.RasterizerState.CullMode = static_cast<D3D12_CULL_MODE>(desc.graphics.rasterizerDesc.cullMode);
    psoDesc.RasterizerState.FrontCounterClockwise = TRUE;
    psoDesc.RasterizerState.DepthBias = desc.graphics.rasterizerDesc.depthBias;
    psoDesc.RasterizerState.DepthBiasClamp = desc.graphics.rasterizerDesc.depthBiasClamp;
    psoDesc.RasterizerState.SlopeScaledDepthBias = desc.graphics.rasterizerDesc.slopeScaledDepthBias;
    psoDesc.RasterizerState.DepthClipEnable = TRUE;
    psoDesc.RasterizerState.MultisampleEnable = FALSE;
    psoDesc.RasterizerState.AntialiasedLineEnable = FALSE;
    psoDesc.RasterizerState.ForcedSampleCount = 0;
    psoDesc.RasterizerState.ConservativeRaster = desc.graphics.rasterizerDesc.conservativeRasterization ? D3D12_CONSERVATIVE_RASTERIZATION_MODE_ON : D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;

    psoDesc.BlendState.AlphaToCoverageEnable = FALSE;
    psoDesc.BlendState.IndependentBlendEnable = FALSE;
    psoDesc.BlendState.RenderTarget[0].BlendEnable = desc.graphics.blendDesc.blendEnable ? TRUE : FALSE;
    psoDesc.BlendState.RenderTarget[0].LogicOpEnable = desc.graphics.blendDesc.logicOpEnable ? TRUE : FALSE;
    psoDesc.BlendState.RenderTarget[0].SrcBlend = static_cast<D3D12_BLEND>(desc.graphics.blendDesc.srcColorBlend);
    psoDesc.BlendState.RenderTarget[0].DestBlend = static_cast<D3D12_BLEND>(desc.graphics.blendDesc.dstColorBlend);
    psoDesc.BlendState.RenderTarget[0].BlendOp = static_cast<D3D12_BLEND_OP>(desc.graphics.blendDesc.blendColorOp);
    psoDesc.BlendState.RenderTarget[0].SrcBlendAlpha = static_cast<D3D12_BLEND>(desc.graphics.blendDesc.srcAlphaBlend);
    psoDesc.BlendState.RenderTarget[0].DestBlendAlpha = static_cast<D3D12_BLEND>(desc.graphics.blendDesc.dstAlphaBlend);
    psoDesc.BlendState.RenderTarget[0].BlendOpAlpha = static_cast<D3D12_BLEND_OP>(desc.graphics.blendDesc.blendAlphaOp);
    psoDesc.BlendState.RenderTarget[0].LogicOp = static_cast<D3D12_LOGIC_OP>(desc.graphics.blendDesc.logicOp);
    psoDesc.BlendState.RenderTarget[0].RenderTargetWriteMask = static_cast<UINT8>(desc.graphics.blendDesc.colorMask);
    psoDesc.DepthStencilState.DepthEnable = desc.graphics.depthStencilDesc.depthTest ? TRUE : FALSE;
    psoDesc.DepthStencilState.DepthWriteMask = desc.graphics.depthStencilDesc.depthMask ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
    psoDesc.DepthStencilState.DepthFunc = static_cast<D3D12_COMPARISON_FUNC>(desc.graphics.depthStencilDesc.depthFunc);
    psoDesc.DepthStencilState.StencilEnable = desc.graphics.depthStencilDesc.stencilTest ? TRUE : FALSE;
    psoDesc.DepthStencilState.StencilReadMask = desc.graphics.depthStencilDesc.stencilReadMask;
    psoDesc.DepthStencilState.StencilWriteMask = desc.graphics.depthStencilDesc.stencilWriteMask;
    psoDesc.DepthStencilState.FrontFace.StencilFailOp = static_cast<D3D12_STENCIL_OP>(desc.graphics.depthStencilDesc.stencilFailOp);
    psoDesc.DepthStencilState.FrontFace.StencilDepthFailOp = static_cast<D3D12_STENCIL_OP>(desc.graphics.depthStencilDesc.stencilDepthFailOp);
    psoDesc.DepthStencilState.FrontFace.StencilPassOp = static_cast<D3D12_STENCIL_OP>(desc.graphics.depthStencilDesc.stencilPassOp);
    psoDesc.DepthStencilState.FrontFace.StencilFunc = static_cast<D3D12_COMPARISON_FUNC>(desc.graphics.depthStencilDesc.stencilFunc);
    psoDesc.DepthStencilState.BackFace.StencilFailOp = static_cast<D3D12_STENCIL_OP>(desc.graphics.depthStencilDesc.stencilFailOp);
    psoDesc.DepthStencilState.BackFace.StencilDepthFailOp = static_cast<D3D12_STENCIL_OP>(desc.graphics.depthStencilDesc.stencilDepthFailOp);
    psoDesc.DepthStencilState.BackFace.StencilPassOp = static_cast<D3D12_STENCIL_OP>(desc.graphics.depthStencilDesc.stencilPassOp);
    psoDesc.DepthStencilState.BackFace.StencilFunc = static_cast<D3D12_COMPARISON_FUNC>(desc.graphics.depthStencilDesc.stencilFunc);

    psoDesc.SampleMask = UINT_MAX;
    psoDesc.PrimitiveTopologyType = static_cast<D3D12_PRIMITIVE_TOPOLOGY_TYPE>(desc.graphics.primitiveTopologyType);
    assert(desc.graphics.numRenderTargets <= MAX_NUM_MRTS);
    psoDesc.NumRenderTargets = desc.graphics.numRenderTargets;
    for(UINT i=0; i<desc.graphics.numRenderTargets; i++)
    {
      psoDesc.RTVFormats[i] = RenderFormat::GetDx12RenderFormat(desc.graphics.rtvFormats[i]).rtvFormat;
    }
    psoDesc.DSVFormat = RenderFormat::GetDx12RenderFormat(desc.graphics.dsvFormat).rtvFormat;
    psoDesc.SampleDesc.Count = 1;
    psoDesc.SampleDesc.Quality = 0;
    psoDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

    if(FAILED(Demo::renderer->GetDevice()->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pipelineState))))
    {
      return false;
    }
  }
  else
  {
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = desc.rootSignature->GetRootSignature();
    psoDesc.CS = desc.graphics.shader->GetByteCode(COMPUTE_SHADER);
    psoDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

    if(FAILED(Demo::renderer->GetDevice()->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pipelineState))))
    {
      return false;
    }
  }

#ifdef _DEBUG
  wchar_t wcharName[DEMO_MAX_STRING];
  swprintf(wcharName, DEMO_MAX_STRING - 1, L"Pipeline state: %hs", name);
  pipelineState->SetName(wcharName);
#endif

  return true;
}
