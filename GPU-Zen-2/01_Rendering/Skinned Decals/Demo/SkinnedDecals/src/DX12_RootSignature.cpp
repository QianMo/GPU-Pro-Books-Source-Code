#include <stdafx.h>
#include <Demo.h>
#include <DX12_RootSignature.h>

bool DX12_RootSignature::Create(const RootSignatureDesc &desc, const char *name)
{
  memcpy(&this->desc, &desc, sizeof(RootSignatureDesc));

  D3D12_ROOT_SIGNATURE_DESC rootSignatureDesc = {};
  rootSignatureDesc.NumParameters = desc.numRootParamDescs;
  assert(desc.numRootParamDescs <= MAX_NUM_ROOT_PARAMS);
  rootSignatureDesc.NumStaticSamplers = desc.numSamplerDescs;
  assert(desc.numSamplerDescs <= MAX_NUM_STATIC_SAMPLERS);
  rootSignatureDesc.pStaticSamplers = (desc.numSamplerDescs > 0) ? reinterpret_cast<const D3D12_STATIC_SAMPLER_DESC*>(desc.samplerDescs) : nullptr;
  rootSignatureDesc.Flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(desc.flags);

  D3D12_ROOT_PARAMETER params[MAX_NUM_ROOT_PARAMS];
  for(UINT i=0; i<desc.numRootParamDescs; i++)
  {
    params[i].ParameterType = static_cast<D3D12_ROOT_PARAMETER_TYPE>(desc.rootParamDescs[i].rootParamType);
    params[i].ShaderVisibility = static_cast<D3D12_SHADER_VISIBILITY>(desc.rootParamDescs[i].shaderVisibility);
    switch(desc.rootParamDescs[i].rootParamType)
    {
    case DESC_TABLE_ROOT_PARAM:
      params[i].DescriptorTable.NumDescriptorRanges = desc.rootParamDescs[i].rootDescTableDesc.numRanges;
      params[i].DescriptorTable.pDescriptorRanges = reinterpret_cast<const D3D12_DESCRIPTOR_RANGE*>(desc.rootParamDescs[i].rootDescTableDesc.ranges);
      break;

    case CONST_ROOT_PARAM:
      assert((desc.rootParamDescs[i].rootConstDesc.numConsts > 0) && (desc.rootParamDescs[i].rootConstDesc.numConsts <= MAX_NUM_ROOT_CONSTS));
      memcpy(&params[i].Constants, &desc.rootParamDescs[i].rootConstDesc, sizeof(D3D12_ROOT_CONSTANTS));
      break;

    case CBV_ROOT_PARAM:
    case SRV_ROOT_PARAM:
    case UAV_ROOT_PARAM:
      memcpy(&params[i].Descriptor, &desc.rootParamDescs[i].rootDesc, sizeof(D3D12_ROOT_DESCRIPTOR));
    }
  }
  rootSignatureDesc.pParameters = (desc.numRootParamDescs > 0) ? params : nullptr;

  ComPtr<ID3DBlob> signatureByteCode;
  ComPtr<ID3DBlob> errorMsg;
  if(FAILED(D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signatureByteCode, &errorMsg)))
  {
    if(errorMsg)
    {
      char errorTitle[512];
      wsprintf(errorTitle, "Root signature error");
      MessageBox(nullptr, (char*)errorMsg->GetBufferPointer(), errorTitle, MB_OK | MB_ICONEXCLAMATION);
    }
  }

  if(FAILED(Demo::renderer->GetDevice()->CreateRootSignature(0, signatureByteCode->GetBufferPointer(), signatureByteCode->GetBufferSize(), IID_PPV_ARGS(&rootSignature))))
  {
    return false;
  }

#ifdef _DEBUG
  wchar_t wcharName[DEMO_MAX_STRING];
  swprintf(wcharName, DEMO_MAX_STRING - 1, L"Root signature: %hs", name);
  rootSignature->SetName(wcharName);
#endif

  return true;
}