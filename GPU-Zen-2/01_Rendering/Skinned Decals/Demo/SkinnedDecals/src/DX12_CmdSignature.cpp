#include <stdafx.h>
#include <Demo.h>
#include <DX12_CmdSignature.h>

bool DX12_CmdSignature::Create(const CmdSignatureDesc &desc, const char *name)
{
  memcpy(&this->desc, &desc, sizeof(CmdSignatureDesc));
 
  D3D12_INDIRECT_ARGUMENT_DESC argDescs[MAX_NUM_CMD_SIGNATURE_ARGS];
  for(UINT i = 0; i < desc.numArgDescs; i++)
  {
    argDescs[i].Type = static_cast<D3D12_INDIRECT_ARGUMENT_TYPE>(desc.argDescs[i].argType);
    switch(desc.argDescs[i].argType)
    {
    case VERTEX_BUFFER_INDIRECT_ARG_TYPE:
      argDescs[i].VertexBuffer.Slot = desc.argDescs[i].vertexBufferDesc.slot;
      break;
    case CONST_INDIRECT_ARG_TYPE:
      argDescs[i].Constant.RootParameterIndex = desc.argDescs[i].constDesc.rootParamIndex;
      argDescs[i].Constant.Num32BitValuesToSet = desc.argDescs[i].constDesc.numConsts;
      argDescs[i].Constant.DestOffsetIn32BitValues = 0;
      break;
    case CONST_BUFFER_INDIRECT_ARG_TYPE:
      argDescs[i].ConstantBufferView.RootParameterIndex = desc.argDescs[i].constBufferDesc.rootParamIndex;
      break;
    case SRV_INDIRECT_ARG_TYPE:
      argDescs[i].ShaderResourceView.RootParameterIndex = desc.argDescs[i].srvDesc.rootParamIndex;
      break;
    case UAV_INDIRECT_ARG_TYPE:
      argDescs[i].UnorderedAccessView.RootParameterIndex = desc.argDescs[i].uavDesc.rootParamIndex;
      break;
    }
  }

  D3D12_COMMAND_SIGNATURE_DESC cmdSignatureDesc = {};
  cmdSignatureDesc.pArgumentDescs = argDescs;
  cmdSignatureDesc.NumArgumentDescs = desc.numArgDescs;
  cmdSignatureDesc.ByteStride = desc.argStride;

  if(FAILED(Demo::renderer->GetDevice()->CreateCommandSignature(&cmdSignatureDesc,
    desc.rootSignature ? desc.rootSignature->GetRootSignature() : nullptr, IID_PPV_ARGS(&cmdSignature))))
  {
    return false;
  }

#ifdef _DEBUG
  wchar_t wcharName[DEMO_MAX_STRING];
  swprintf(wcharName, DEMO_MAX_STRING - 1, L"Command signature: %hs", name);
  cmdSignature->SetName(wcharName);
#endif

  return true;
}