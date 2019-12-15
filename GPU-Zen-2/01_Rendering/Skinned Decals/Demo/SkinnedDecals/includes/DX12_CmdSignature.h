#ifndef DX12_CMD_SIGNATURE_H
#define DX12_CMD_SIGNATURE_H

#include <render_states.h>
#include <DX12_RootSignature.h>

enum indirectArgTypes
{
  DRAW_INDIRECT_ARG_TYPE          = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW,
  DRAW_INDEXED_INDIRECT_ARG_TYPE  = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED,
  DISPATCH_INDIRECT_ARG_TYPE      = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
  VERTEX_BUFFER_INDIRECT_ARG_TYPE = D3D12_INDIRECT_ARGUMENT_TYPE_VERTEX_BUFFER_VIEW,
  INDEX_BUFFER_INDIRECT_ARG_TYPE  = D3D12_INDIRECT_ARGUMENT_TYPE_INDEX_BUFFER_VIEW,
  CONST_INDIRECT_ARG_TYPE         = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT,
  CONST_BUFFER_INDIRECT_ARG_TYPE  = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT_BUFFER_VIEW,
  SRV_INDIRECT_ARG_TYPE           = D3D12_INDIRECT_ARGUMENT_TYPE_SHADER_RESOURCE_VIEW,
  UAV_INDIRECT_ARG_TYPE           = D3D12_INDIRECT_ARGUMENT_TYPE_UNORDERED_ACCESS_VIEW
};

struct CmdSignatureArgDesc
{
  CmdSignatureArgDesc()
  {
    memset(this, 0, sizeof(CmdSignatureArgDesc));
  }

  bool operator== (const CmdSignatureArgDesc &desc) const
  {
    if(argType != desc.argType)
    {
      return false;
    }
    switch(argType)
    {
    case VERTEX_BUFFER_INDIRECT_ARG_TYPE:
      if(vertexBufferDesc.slot != desc.vertexBufferDesc.slot)
        return false;
      break;
    case CONST_INDIRECT_ARG_TYPE:
      if((constDesc.rootParamIndex != desc.constDesc.rootParamIndex) || (constDesc.numConsts != desc.constDesc.numConsts))
        return false;
      break;
    case CONST_BUFFER_INDIRECT_ARG_TYPE:
      if(constBufferDesc.rootParamIndex != desc.constBufferDesc.rootParamIndex)
        return false;
      break;
    case SRV_INDIRECT_ARG_TYPE:
      if(srvDesc.rootParamIndex != desc.srvDesc.rootParamIndex)
        return false;
      break;
    case UAV_INDIRECT_ARG_TYPE:
      if(uavDesc.rootParamIndex != desc.uavDesc.rootParamIndex)
        return false;
      break;
    }
    return true;
  }

  bool operator!= (const CmdSignatureArgDesc &desc) const
  {
    return !((*this) == desc);
  }

  indirectArgTypes argType;
  union
  {
    struct
    {
      UINT slot;
    } vertexBufferDesc;

    struct
    {
      UINT rootParamIndex;
      UINT numConsts;
    } constDesc;

    struct
    {
      UINT rootParamIndex;
    } constBufferDesc;

    struct
    {
      UINT rootParamIndex;
    } srvDesc;

    struct
    {
      UINT rootParamIndex;
    } uavDesc;
  };
};

struct CmdSignatureDesc
{
  CmdSignatureDesc() :
    argStride(0),
    numArgDescs(0),
    rootSignature(nullptr)
  {
  }

  bool operator== (const CmdSignatureDesc &desc) const
  {
    if((rootSignature && (!desc.rootSignature)) || ((!rootSignature) && desc.rootSignature))
      return false;
    if(rootSignature && desc.rootSignature && (rootSignature->GetDesc() != desc.rootSignature->GetDesc()))
      return false;
    if((argStride != desc.argStride) || (numArgDescs != desc.numArgDescs))
    {
      return false;
    }
    for(UINT i=0; i<numArgDescs; i++)
    {
      if(argDescs[i] != desc.argDescs[i])
        return false;
    }
    return true;
  }

  bool operator!= (const CmdSignatureDesc &desc) const
  {
    return !((*this) == desc);
  }

  UINT argStride;
  UINT numArgDescs;
  CmdSignatureArgDesc argDescs[MAX_NUM_CMD_SIGNATURE_ARGS];
  DX12_RootSignature *rootSignature;
};

// DX12_CmdSignature
//
class DX12_CmdSignature
{
public:
  bool Create(const CmdSignatureDesc &desc, const char *name);

  ID3D12CommandSignature* GetCmdSignature() const
  {
    return cmdSignature.Get();
  }

  const CmdSignatureDesc& GetDesc() const
  {
    return desc;
  }

private:
  ComPtr<ID3D12CommandSignature> cmdSignature;
  CmdSignatureDesc desc;

};

#endif 