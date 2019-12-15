#ifndef DX12_PIPELINE_STATE_H
#define DX12_PIPELINE_STATE_H

#include <render_states.h>
#include <Image.h>

enum pipelineStateTypes
{
  GRAPHICS_PIPELINE_STATE=0,
  COMPUTE_PIPELINE_STATE
};

enum inputElements
{
  POSITION_INPUT_ELEMENT=0,
  TEXCOORDS_INPUT_ELEMENT,
  NORMAL_INPUT_ELEMENT,
  TANGENT_INPUT_ELEMENT,
  COLOR_INPUT_ELEMENT
};

class DX12_RootSignature;
class DX12_Shader;

struct InputElementDesc
{
  bool operator== (const InputElementDesc &desc) const
  {
    return ((inputElement == desc.inputElement) && (format == desc.format) && (offset == desc.offset) && 
            (vertexBufferSlot == desc.vertexBufferSlot) && (instanceDataStepRate == desc.instanceDataStepRate));
  }

  bool operator!= (const InputElementDesc &desc) const
  {
    return !((*this) == desc);
  }

  inputElements inputElement;
  renderFormats format;
  UINT offset;
  UINT vertexBufferSlot;
  UINT instanceDataStepRate;
};

struct InputLayout
{
  bool operator== (const InputLayout &desc) const
  {
    if(numElementDescs != desc.numElementDescs)
      return false;
    for(UINT i=0; i<numElementDescs; i++)
    {
      if(elementDescs[i] != desc.elementDescs[i])
        return false;
    }
    return true;
  }

  bool operator!= (const InputLayout &desc) const
  {
    return !((*this) == desc);
  }

  InputElementDesc elementDescs[MAX_NUM_INPUT_ELEMENT_DESCS];
  UINT numElementDescs;
};

struct BlendDesc
{
  void Reset()
  {
    srcColorBlend = ONE_BLEND;
    dstColorBlend = ONE_BLEND;
    blendColorOp = ADD_BLEND_OP;
    srcAlphaBlend = ONE_BLEND;
    dstAlphaBlend = ONE_BLEND;
    blendAlphaOp = ADD_BLEND_OP;
    logicOp = NOOP_LOGIC_OP;
    blendEnable = false;
    logicOpEnable = false;
    colorMask = ALL_COLOR_MASK;
  }

  bool operator== (const BlendDesc &desc) const
  {
    return((srcColorBlend == desc.srcColorBlend) && (dstColorBlend == desc.dstColorBlend) && (blendColorOp == desc.blendColorOp) &&
           (srcAlphaBlend == desc.srcAlphaBlend) && (dstAlphaBlend == desc.dstAlphaBlend) && (blendAlphaOp == desc.blendAlphaOp) &&
           (logicOp == desc.logicOp) && (blendEnable == desc.blendEnable) && (logicOpEnable == desc.logicOpEnable) &&
           (colorMask == desc.colorMask));
  }

  bool operator!= (const BlendDesc &desc) const
  {
    return !((*this) == desc);
  }

  blendStates srcColorBlend;
  blendStates dstColorBlend;
  blendOps blendColorOp;
  blendStates srcAlphaBlend;
  blendStates dstAlphaBlend;
  blendOps blendAlphaOp;
  logicOps logicOp;
  bool blendEnable;
  bool logicOpEnable;
  colorMaskBits colorMask;
};

struct RasterizerDesc
{
  void Reset()
  {
    fillMode = SOLID_FILL;
    cullMode = NONE_CULL;
    depthBias = 0;
    depthBiasClamp = 0.0f;
    slopeScaledDepthBias = 0.0f;
    conservativeRasterization = false;
  }

  bool operator== (const RasterizerDesc &desc) const
  {
    return ((fillMode == desc.fillMode) && (cullMode == desc.cullMode) && (depthBias == desc.depthBias) &&
            (conservativeRasterization == desc.conservativeRasterization) && IS_EQUAL(depthBiasClamp, desc.depthBiasClamp) &&
            IS_EQUAL(slopeScaledDepthBias, desc.slopeScaledDepthBias));
  }

  bool operator!= (const RasterizerDesc &desc) const
  {
    return !((*this) == desc);
  }

  fillModes fillMode;
  cullModes cullMode;
  int depthBias;
  float depthBiasClamp;
  float slopeScaledDepthBias;
  bool conservativeRasterization;
};

struct DepthStencilDesc
{
  void Reset()
  {
    depthFunc = LEQUAL_CMP_FUNC;
    stencilFunc = ALWAYS_CMP_FUNC;
    stencilFailOp = KEEP_STENCIL_OP;
    stencilDepthFailOp = INCR_SAT_STENCIL_OP;
    stencilPassOp = INCR_SAT_STENCIL_OP;
    depthTest = true;
    depthMask = true;
    stencilTest = false;
    stencilReadMask = 0xff;
    stencilWriteMask = 0xff;
  }

  bool operator== (const DepthStencilDesc &desc) const
  {
    return ((depthFunc == desc.depthFunc) && (stencilFunc == desc.stencilFunc) && (stencilFailOp == desc.stencilFailOp) &&
            (stencilDepthFailOp == desc.stencilDepthFailOp) && (stencilPassOp == desc.stencilPassOp) && (depthTest == desc.depthTest) &&
            (depthMask == desc.depthMask) && (stencilTest == desc.stencilTest) && (stencilReadMask == desc.stencilReadMask) &&
            (stencilWriteMask == desc.stencilWriteMask));
  }

  bool operator!= (const DepthStencilDesc &desc) const
  {
    return !((*this) == desc);
  }

  comparisonFuncs depthFunc;
  comparisonFuncs stencilFunc;
  stencilOps stencilFailOp;
  stencilOps stencilDepthFailOp;
  stencilOps stencilPassOp;
  bool depthTest;
  bool depthMask;
  bool stencilTest;
  unsigned char stencilReadMask;
  unsigned char stencilWriteMask;
};

struct GraphicsPipelineStateDesc
{
  void Reset()
  {
    memset(this, 0, sizeof(GraphicsPipelineStateDesc));
    primitiveTopologyType = TRIANGLE_PRIMITIVE_TOPOLOGY_TYPE;
    blendDesc.Reset();
    rasterizerDesc.Reset();
    depthStencilDesc.Reset();
  }

  bool operator== (const GraphicsPipelineStateDesc &desc) const
  {
    if((shader != desc.shader) || (numRenderTargets != desc.numRenderTargets) || (dsvFormat != desc.dsvFormat) ||
       (primitiveTopologyType != desc.primitiveTopologyType) || (inputLayout != desc.inputLayout) || (blendDesc != desc.blendDesc) ||
       (rasterizerDesc != desc.rasterizerDesc) || (depthStencilDesc != desc.depthStencilDesc))
    {
      return false;
    }
    for(UINT i=0; i<numRenderTargets; i++)
    {
      if(rtvFormats[i] != desc.rtvFormats[i])
        return false;
    }
    return true;
  }

  bool operator!= (const GraphicsPipelineStateDesc &desc) const
  {
    return !((*this) == desc);
  }

  DX12_Shader *shader;
  UINT numRenderTargets;
  renderFormats rtvFormats[MAX_NUM_MRTS];
  renderFormats dsvFormat;
  primitiveTopologyTypes primitiveTopologyType;
  InputLayout inputLayout;
  BlendDesc blendDesc;
  RasterizerDesc rasterizerDesc;
  DepthStencilDesc depthStencilDesc;
};

struct ComputePipelineStateDesc
{
  void Reset()
  {
    shader = nullptr;
  }

  bool operator== (const ComputePipelineStateDesc &desc) const
  {
    return (shader == desc.shader);
  }

  bool operator!= (const ComputePipelineStateDesc &desc) const
  {
    return !((*this) == desc);
  }

  DX12_Shader *shader;
};

struct PipelineStateDesc
{
public:
  friend class DX12_PipelineState;

  explicit PipelineStateDesc(pipelineStateTypes pipelineStateType) :
    rootSignature(nullptr)
  {
    this->pipelineStateType = pipelineStateType;
    if(pipelineStateType == GRAPHICS_PIPELINE_STATE)
      graphics.Reset();
    else
      compute.Reset();
  }

  bool operator== (const PipelineStateDesc &desc) const
  {
    if((rootSignature != desc.rootSignature) || (pipelineStateType != desc.pipelineStateType))
    {
      return false;
    }
    if(pipelineStateType == GRAPHICS_PIPELINE_STATE)
    {
      if(graphics != desc.graphics)
        return false;
    }
    else
    {
      if(compute != desc.compute)
        return false;
    }
    return true;
  }

  bool operator!= (const PipelineStateDesc &desc) const
  {
    return !((*this) == desc);
  }

  DX12_RootSignature *rootSignature;
  union
  {
    GraphicsPipelineStateDesc graphics;
    ComputePipelineStateDesc compute;
  };

private:
  pipelineStateTypes pipelineStateType;
};

// DX12_PipelineState
//
class DX12_PipelineState
{
public:
  DX12_PipelineState() :
    desc(GRAPHICS_PIPELINE_STATE)
  {
  }

  bool Create(const PipelineStateDesc &desc, const char *name);

  ID3D12PipelineState* GetPipelineState() const
  {
    return pipelineState.Get();
  }

  const PipelineStateDesc& GetDesc() const
  {
    return desc;
  }

private:
  ComPtr<ID3D12PipelineState> pipelineState;
  PipelineStateDesc desc;

};

#endif
