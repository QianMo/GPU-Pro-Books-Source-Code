#include <stdafx.h>
#include <Demo.h>
#include <SkinnedDecals.h>

bool SkinnedDecals::Init(const DemoModel *parentModel, const char** materialNames, UINT numMaterials)
{
  if((!parentModel) || (numMaterials > MAX_NUM_DECAL_MATERIALS))
  {
    return false;
  }
  this->parentModel = parentModel;
  numDecalMaterials = numMaterials;

  {
    // create decal lookup textures
    for(UINT i=0; i<parentModel->numSubModels; i++)
    {
      DemoSubModel &subModel = parentModel->subModels[i];
      if(subModel.material->receiveDecals)
      {
        TextureDesc desc;
        desc.width = subModel.material->textures[COLOR_TEX_ID]->GetTextureDesc().width;
        desc.height = subModel.material->textures[COLOR_TEX_ID]->GetTextureDesc().height;
        desc.format = Demo::renderer->GetDeviceFeatureSupport().typedUavLoads ? RGBA8_RENDER_FORMAT : R32UI_RENDER_FORMAT;
        desc.flags = UAV_TEXTURE_FLAG;
        desc.initResourceState = UNORDERED_ACCESS_RESOURCE_STATE;
        decalLookupTextures[numDecalLookupTextures] = Demo::resourceManager->CreateTexture(desc, "Decal lookup map");
        if(!decalLookupTextures[numDecalLookupTextures])
          return false;
        subModel.materialDT.AddTextureSrv(decalLookupTextures[numDecalLookupTextures]);

        decalConstData.subModelInfos[numDecalLookupTextures].firstIndex = subModel.firstIndex;
        decalConstData.subModelInfos[numDecalLookupTextures].numTris = subModel.numIndices / 3;
        decalConstData.subModelInfos[numDecalLookupTextures].decalLookupMapWidth = static_cast<float>(desc.width);
        decalConstData.subModelInfos[numDecalLookupTextures].decalLookupMapHeight = static_cast<float>(desc.height);
        numDecalLookupTextures++;
      }
    }
  }

  {
    // create dummy render-target/ viewport set/ scissor rectangle set for filling decal lookup map
    UINT width = 0;
    UINT height = 0;
    for(UINT i=0; i<numDecalLookupTextures; i++)
    {
      width = max(width, decalLookupTextures[i]->GetTextureDesc().width);
      height = max(height, decalLookupTextures[i]->GetTextureDesc().height);
    }
    decalConstData.decalLookupRtWidth = static_cast<float>(width);
    decalConstData.decalLookupRtHeight = static_cast<float>(height);

    if((width > 0) && (height > 0))
    {
      TextureDesc desc;
      desc.width = width;
      desc.height = height;
      desc.format = R8_RENDER_FORMAT;
      desc.flags = RENDER_TARGET_TEXTURE_FLAG;
      desc.initResourceState = PIXEL_SHADER_RESOURCE_RESOURCE_STATE;
      decalLookupMapRT = Demo::renderer->CreateRenderTarget(desc, "Decal lookup map render target");
      if(!decalLookupMapRT)
        return false;

      Viewport viewport;
      viewport.width = decalConstData.decalLookupRtWidth;
      viewport.height = decalConstData.decalLookupRtHeight;
      decalLookupMapVPS = Demo::renderer->CreateViewportSet(&viewport, 1);
      if(!decalLookupMapVPS)
        return false;

      ScissorRect scissorRect;
      scissorRect.right = width;
      scissorRect.bottom = height;
      decalLookupMapSRS = Demo::renderer->CreateScissorRectSet(&scissorRect, 1);
      if(!decalLookupMapSRS)
        return false;
    }
  }

  {
    // create decal validity structured buffer
    BufferDesc desc;
    desc.bufferType = STRUCTURED_BUFFER;
    desc.elementSize = sizeof(UINT);
    desc.numElements = MAX_NUM_SKINNED_DECALS;
    desc.flags = DYNAMIC_BUFFER_FLAG;
    decalValiditySB = Demo::renderer->CreateBuffer(desc, "Decal validity buffer");
    if(!decalValiditySB)  
      return false;
  }

  {
    // create structured buffer for storing decal info
    BufferDesc desc;
    desc.bufferType = STRUCTURED_BUFFER;
    desc.elementSize = sizeof(DecalInfo);
    desc.numElements = 1;
    desc.flags = DYNAMIC_BUFFER_FLAG;
    decalInfoSB = Demo::renderer->CreateBuffer(desc, "DecalInfo");
    if(!decalInfoSB)  
      return false;
  }

  {
    // create constant buffer for storing decal const data
    BufferDesc bufferDesc;
    bufferDesc.bufferType = CONSTANT_BUFFER;
    bufferDesc.elementSize = sizeof(DecalConstData);
    bufferDesc.numElements = 1;
    bufferDesc.flags = CPU_WRITE_BUFFER_FLAG | DYNAMIC_BUFFER_FLAG;
    decalCB = Demo::renderer->CreateBuffer(bufferDesc, "DecalConstData");
    if(!decalCB)
      return false;
  }

  {
    // create pipeline state for clearing decal lookup map
    RootSignatureDesc rootSignatureDesc;
    // decal mesh info root constant
    rootSignatureDesc.rootParamDescs[0].rootParamType = CONST_ROOT_PARAM;
    rootSignatureDesc.rootParamDescs[0].rootConstDesc.numConsts = 1;
    rootSignatureDesc.rootParamDescs[0].rootConstDesc.shaderReg = 0;
    rootSignatureDesc.rootParamDescs[0].rootConstDesc.regSpace = 0;
    rootSignatureDesc.rootParamDescs[0].shaderVisibility = CS_SHADER_VIS;
    // model info constant buffer
    rootSignatureDesc.rootParamDescs[1].rootParamType = CBV_ROOT_PARAM;
    rootSignatureDesc.rootParamDescs[1].rootDesc.shaderReg = 1;
    rootSignatureDesc.rootParamDescs[1].rootDesc.regSpace = 0;
    rootSignatureDesc.rootParamDescs[1].shaderVisibility = CS_SHADER_VIS;
    // decal lookup map UAVs
    rootSignatureDesc.rootParamDescs[2].rootParamType = DESC_TABLE_ROOT_PARAM;
    rootSignatureDesc.rootParamDescs[2].rootDescTableDesc.numRanges = 1;
    rootSignatureDesc.rootParamDescs[2].rootDescTableDesc.ranges[0].rangeType = UAV_ROOT_DESC_TABLE_RANGE;
    rootSignatureDesc.rootParamDescs[2].rootDescTableDesc.ranges[0].numDescs = MAX_NUM_SUBMODELS;
    rootSignatureDesc.rootParamDescs[2].rootDescTableDesc.ranges[0].baseShaderReg = 0;
    rootSignatureDesc.rootParamDescs[2].rootDescTableDesc.ranges[0].regSpace = 0;
    rootSignatureDesc.rootParamDescs[2].shaderVisibility = CS_SHADER_VIS;
    rootSignatureDesc.numRootParamDescs = 3;
    DX12_RootSignature *rootSignature = Demo::renderer->CreateRootSignature(rootSignatureDesc, "Clear decal lookup map");
    if(!rootSignature)
      return false;

    UINT permutationMask = Demo::renderer->GetDeviceFeatureSupport().typedUavLoads ? 1 : 0; // permutation 1 = TYPED_UAV_LOADS
    DX12_Shader *shader = Demo::resourceManager->LoadShader("shaders/clearDecalLookupMap.sdr", permutationMask);
    if(!shader)
      return false;
 
    PipelineStateDesc pipelineStateDesc(COMPUTE_PIPELINE_STATE);
    pipelineStateDesc.rootSignature = rootSignature;
    pipelineStateDesc.compute.shader = shader;
    clearDecalLookupMapPS = Demo::renderer->CreatePipelineState(pipelineStateDesc, "Clear decal lookup map");
    if(!clearDecalLookupMapPS)
      return false;
  }

  {
    // create pipeline state for clearing decal validity buffer
    RootSignatureDesc rootSignatureDesc;
    rootSignatureDesc.rootParamDescs[0].rootParamType = UAV_ROOT_PARAM;
    rootSignatureDesc.rootParamDescs[0].rootDesc.shaderReg = 0;
    rootSignatureDesc.rootParamDescs[0].rootDesc.regSpace = 0;
    rootSignatureDesc.rootParamDescs[0].shaderVisibility = CS_SHADER_VIS;
    rootSignatureDesc.numRootParamDescs = 1;
    DX12_RootSignature *rootSignature = Demo::renderer->CreateRootSignature(rootSignatureDesc, "Clear decal validity buffer");
    if(!rootSignature)
      return false;

    DX12_Shader *shader = Demo::resourceManager->LoadShader("shaders/clearDecalValidityBuffer.sdr");
    if(!shader)
      return false;

    PipelineStateDesc pipelineStateDesc(COMPUTE_PIPELINE_STATE);
    pipelineStateDesc.rootSignature = rootSignature;
    pipelineStateDesc.compute.shader = shader;
    clearDecalValidityBufferPS = Demo::renderer->CreatePipelineState(pipelineStateDesc, "Clear decal validity buffer");
    if(!clearDecalValidityBufferPS)
      return false;
  }

  {
    // create pipeline state for clearing decal info
    RootSignatureDesc rootSignatureDesc;
    rootSignatureDesc.rootParamDescs[0].rootParamType = DESC_TABLE_ROOT_PARAM;
    rootSignatureDesc.rootParamDescs[0].rootDescTableDesc.numRanges = 1;
    rootSignatureDesc.rootParamDescs[0].rootDescTableDesc.ranges[0].rangeType = UAV_ROOT_DESC_TABLE_RANGE;
    rootSignatureDesc.rootParamDescs[0].rootDescTableDesc.ranges[0].numDescs = 1;
    rootSignatureDesc.rootParamDescs[0].rootDescTableDesc.ranges[0].baseShaderReg = 0;
    rootSignatureDesc.rootParamDescs[0].rootDescTableDesc.ranges[0].regSpace = 0;
    rootSignatureDesc.rootParamDescs[0].shaderVisibility = CS_SHADER_VIS;
    rootSignatureDesc.numRootParamDescs = 1;
    DX12_RootSignature *rootSignature = Demo::renderer->CreateRootSignature(rootSignatureDesc, "Clear DecalInfo");
    if(!rootSignature)
      return false;

    DX12_Shader *shader = Demo::resourceManager->LoadShader("shaders/clearDecalInfo.sdr");
    if(!shader)
      return false;
 
    PipelineStateDesc pipelineStateDesc(COMPUTE_PIPELINE_STATE);
    pipelineStateDesc.rootSignature = rootSignature;
    pipelineStateDesc.compute.shader = shader;
    clearDecalInfoPS = Demo::renderer->CreatePipelineState(pipelineStateDesc, "Clear DecalInfo");
    if(!clearDecalInfoPS)
      return false;
  }

  {
    // create pipeline state for finding closest intersection
    RootSignatureDesc rootSignatureDesc;
    // decal mesh info root constant
    rootSignatureDesc.rootParamDescs[0].rootParamType = CONST_ROOT_PARAM;
    rootSignatureDesc.rootParamDescs[0].rootConstDesc.numConsts = 1;
    rootSignatureDesc.rootParamDescs[0].rootConstDesc.shaderReg = 0;
    rootSignatureDesc.rootParamDescs[0].rootConstDesc.regSpace = 0;
    rootSignatureDesc.rootParamDescs[0].shaderVisibility = CS_SHADER_VIS;
    // model info constant buffer
    rootSignatureDesc.rootParamDescs[1].rootParamType = CBV_ROOT_PARAM;
    rootSignatureDesc.rootParamDescs[1].rootDesc.shaderReg = 1;
    rootSignatureDesc.rootParamDescs[1].rootDesc.regSpace = 0;
    rootSignatureDesc.rootParamDescs[1].shaderVisibility = CS_SHADER_VIS;
    // vertex position buffer/ index buffer SRVs and decal info buffer UAV
    rootSignatureDesc.rootParamDescs[2].rootParamType = DESC_TABLE_ROOT_PARAM;
    rootSignatureDesc.rootParamDescs[2].rootDescTableDesc.numRanges = 2;
    rootSignatureDesc.rootParamDescs[2].rootDescTableDesc.ranges[0].rangeType = SRV_ROOT_DESC_TABLE_RANGE;
    rootSignatureDesc.rootParamDescs[2].rootDescTableDesc.ranges[0].numDescs = 2;
    rootSignatureDesc.rootParamDescs[2].rootDescTableDesc.ranges[0].baseShaderReg = 0;
    rootSignatureDesc.rootParamDescs[2].rootDescTableDesc.ranges[0].regSpace = 0;
    rootSignatureDesc.rootParamDescs[2].rootDescTableDesc.ranges[1].rangeType = UAV_ROOT_DESC_TABLE_RANGE;
    rootSignatureDesc.rootParamDescs[2].rootDescTableDesc.ranges[1].descTableOffset = 2;
    rootSignatureDesc.rootParamDescs[2].rootDescTableDesc.ranges[1].numDescs = 1;
    rootSignatureDesc.rootParamDescs[2].rootDescTableDesc.ranges[1].baseShaderReg = 0;
    rootSignatureDesc.rootParamDescs[2].rootDescTableDesc.ranges[1].regSpace = 0;
    rootSignatureDesc.rootParamDescs[2].shaderVisibility = CS_SHADER_VIS;
    rootSignatureDesc.numRootParamDescs = 3;
    DX12_RootSignature *rootSignature = Demo::renderer->CreateRootSignature(rootSignatureDesc, "Raycast");
    if(!rootSignature)
      return false;

    DX12_Shader *shader = Demo::resourceManager->LoadShader("shaders/raycast.sdr");
    if(!shader)
      return false;
 
    PipelineStateDesc pipelineStateDesc(COMPUTE_PIPELINE_STATE);
    pipelineStateDesc.rootSignature = rootSignature;
    pipelineStateDesc.compute.shader = shader;
    raycastPS = Demo::renderer->CreatePipelineState(pipelineStateDesc, "Raycast");
    if(!raycastPS)
      return false;
  }

  {
    // create pipeline state for calculating decal info
    RootSignatureDesc rootSignatureDesc;
    // model info constant buffer
    rootSignatureDesc.rootParamDescs[0].rootParamType = CBV_ROOT_PARAM;
    rootSignatureDesc.rootParamDescs[0].rootDesc.shaderReg = 0;
    rootSignatureDesc.rootParamDescs[0].rootDesc.regSpace = 0;
    rootSignatureDesc.rootParamDescs[0].shaderVisibility = CS_SHADER_VIS;
    // vertex position buffer/ index buffer SRVs and decal info buffer UAV
    rootSignatureDesc.rootParamDescs[1].rootParamType = DESC_TABLE_ROOT_PARAM;
    rootSignatureDesc.rootParamDescs[1].rootDescTableDesc.numRanges = 2;
    rootSignatureDesc.rootParamDescs[1].rootDescTableDesc.ranges[0].rangeType = SRV_ROOT_DESC_TABLE_RANGE;
    rootSignatureDesc.rootParamDescs[1].rootDescTableDesc.ranges[0].numDescs = 2;
    rootSignatureDesc.rootParamDescs[1].rootDescTableDesc.ranges[0].baseShaderReg = 0;
    rootSignatureDesc.rootParamDescs[1].rootDescTableDesc.ranges[0].regSpace = 0;
    rootSignatureDesc.rootParamDescs[1].rootDescTableDesc.ranges[1].rangeType = UAV_ROOT_DESC_TABLE_RANGE;
    rootSignatureDesc.rootParamDescs[1].rootDescTableDesc.ranges[1].descTableOffset = 2;
    rootSignatureDesc.rootParamDescs[1].rootDescTableDesc.ranges[1].numDescs = 1;
    rootSignatureDesc.rootParamDescs[1].rootDescTableDesc.ranges[1].baseShaderReg = 0;
    rootSignatureDesc.rootParamDescs[1].rootDescTableDesc.ranges[1].regSpace = 0;
    rootSignatureDesc.rootParamDescs[1].shaderVisibility = CS_SHADER_VIS;
    rootSignatureDesc.numRootParamDescs = 2;
    DX12_RootSignature *rootSignature = Demo::renderer->CreateRootSignature(rootSignatureDesc, "Calculate DecalInfo");
    if(!rootSignature)
      return false;

    DX12_Shader *shader = Demo::resourceManager->LoadShader("shaders/calcDecalInfo.sdr");
    if(!shader)
      return false;
 
    PipelineStateDesc pipelineStateDesc(COMPUTE_PIPELINE_STATE);
    pipelineStateDesc.rootSignature = rootSignature;
    pipelineStateDesc.compute.shader = shader;
    calcDecalInfoPS = Demo::renderer->CreatePipelineState(pipelineStateDesc, "Calculate DecalInfo");
    if(!calcDecalInfoPS)
      return false;
  }

  {
    // create pipeline state to render decal into decal lookup map
    RootSignatureDesc rootSignatureDesc;
    rootSignatureDesc.rootParamDescs[0].rootParamType = DESC_TABLE_ROOT_PARAM;
    rootSignatureDesc.rootParamDescs[0].rootDescTableDesc.numRanges = 2;
    rootSignatureDesc.rootParamDescs[0].rootDescTableDesc.ranges[0].rangeType = SRV_ROOT_DESC_TABLE_RANGE;
    rootSignatureDesc.rootParamDescs[0].rootDescTableDesc.ranges[0].numDescs = 4;
    rootSignatureDesc.rootParamDescs[0].rootDescTableDesc.ranges[0].baseShaderReg = 0;
    rootSignatureDesc.rootParamDescs[0].rootDescTableDesc.ranges[0].regSpace = 0;
    rootSignatureDesc.rootParamDescs[0].rootDescTableDesc.ranges[1].rangeType = UAV_ROOT_DESC_TABLE_RANGE;
    rootSignatureDesc.rootParamDescs[0].rootDescTableDesc.ranges[1].descTableOffset = 4;
    rootSignatureDesc.rootParamDescs[0].rootDescTableDesc.ranges[1].numDescs = 2 + MAX_NUM_SUBMODELS;
    rootSignatureDesc.rootParamDescs[0].rootDescTableDesc.ranges[1].baseShaderReg = 1;
    rootSignatureDesc.rootParamDescs[0].rootDescTableDesc.ranges[1].regSpace = 0;
    rootSignatureDesc.rootParamDescs[0].shaderVisibility = ALL_SHADER_VIS;
    rootSignatureDesc.numRootParamDescs = 1;
    DX12_RootSignature *rootSignature = Demo::renderer->CreateRootSignature(rootSignatureDesc, "Render decal");
    if(!rootSignature)
      return false;
    
    UINT permutationMask = Demo::renderer->GetDeviceFeatureSupport().typedUavLoads ? 1 : 0; // permutation 1 = TYPED_UAV_LOADS
    DX12_Shader *shader = Demo::resourceManager->LoadShader("shaders/baseDecal.sdr", permutationMask);
    if(!shader)
      return false;

    PipelineStateDesc pipelineStateDesc(GRAPHICS_PIPELINE_STATE);
    pipelineStateDesc.rootSignature = rootSignature;
    pipelineStateDesc.graphics.shader = shader;
    pipelineStateDesc.graphics.numRenderTargets = 1;
    pipelineStateDesc.graphics.rtvFormats[0] = decalLookupMapRT->GetTexture()->GetTextureDesc().format;
    pipelineStateDesc.graphics.blendDesc.colorMask = NONE_COLOR_MASK;
    pipelineStateDesc.graphics.depthStencilDesc.depthTest = false;
    pipelineStateDesc.graphics.depthStencilDesc.depthMask = false;
    renderDecalPS = Demo::renderer->CreatePipelineState(pipelineStateDesc, "Render decal");
    if(!renderDecalPS)
      return false;
  }

  {
    // create commnad signature to render indirectly decal into decal lookup map
    CmdSignatureDesc cmdSignatureDesc;
    cmdSignatureDesc.argStride = sizeof(DecalInfo);
    cmdSignatureDesc.numArgDescs = 1;
    cmdSignatureDesc.argDescs[0].argType = DRAW_INDEXED_INDIRECT_ARG_TYPE;
    renderDecalCS = Demo::renderer->CreateCmdSignature(cmdSignatureDesc, "Render decal");
    if(!renderDecalCS)
      return false;
  }

  {
    // create pipeline state for removing decal
    RootSignatureDesc rootSignatureDesc;
    // model info constant buffer
    rootSignatureDesc.rootParamDescs[0].rootParamType = CBV_ROOT_PARAM;
    rootSignatureDesc.rootParamDescs[0].rootDesc.shaderReg = 0;
    rootSignatureDesc.rootParamDescs[0].rootDesc.regSpace = 0;
    rootSignatureDesc.rootParamDescs[0].shaderVisibility = CS_SHADER_VIS;
    // vertex position buffer/ vertex uvHandedness buffer/ index buffer / decal info buffer/ decal lookup maps SRVs
    rootSignatureDesc.rootParamDescs[1].rootParamType = DESC_TABLE_ROOT_PARAM;
    rootSignatureDesc.rootParamDescs[1].rootDescTableDesc.numRanges = 1;
    rootSignatureDesc.rootParamDescs[1].rootDescTableDesc.ranges[0].rangeType = SRV_ROOT_DESC_TABLE_RANGE;
    rootSignatureDesc.rootParamDescs[1].rootDescTableDesc.ranges[0].numDescs = 4 + MAX_NUM_SUBMODELS;
    rootSignatureDesc.rootParamDescs[1].rootDescTableDesc.ranges[0].baseShaderReg = 0;
    rootSignatureDesc.rootParamDescs[1].rootDescTableDesc.ranges[0].regSpace = 0;
    rootSignatureDesc.rootParamDescs[1].shaderVisibility = CS_SHADER_VIS;
    // decal validity buffer UAV
    rootSignatureDesc.rootParamDescs[2].rootParamType = UAV_ROOT_PARAM;
    rootSignatureDesc.rootParamDescs[2].rootDesc.shaderReg = 0;
    rootSignatureDesc.rootParamDescs[2].rootDesc.regSpace = 0;
    rootSignatureDesc.rootParamDescs[2].shaderVisibility = CS_SHADER_VIS;
    rootSignatureDesc.numRootParamDescs = 3;
    DX12_RootSignature *rootSignature = Demo::renderer->CreateRootSignature(rootSignatureDesc, "Removing decal");
    if(!rootSignature)
      return false;

    UINT permutationMask = Demo::renderer->GetDeviceFeatureSupport().typedUavLoads ? 1 : 0; // permutation 1 = TYPED_UAV_LOADS
    DX12_Shader *shader = Demo::resourceManager->LoadShader("shaders/removeDecal.sdr", permutationMask);
    if(!shader)
      return false;

    PipelineStateDesc pipelineStateDesc(COMPUTE_PIPELINE_STATE);
    pipelineStateDesc.rootSignature = rootSignature;
    pipelineStateDesc.compute.shader = shader;
    removeDecalPS = Demo::renderer->CreatePipelineState(pipelineStateDesc, "Removing decal");
    if(!removeDecalPS)
      return false;
  }

  // create descriptor table for clearing decal lookup textures
  for(UINT i=0; i<numDecalLookupTextures; i++)
  {
    clearDecalsDT.AddTextureUav(decalLookupTextures[i]);
  }

  // create descriptor table for clearing decal info
  clearDecalInfoDT.AddBufferUav(decalInfoSB);

  // create descriptor table for calculating decal info
  calcDecalInfoDT.AddBufferSrv(parentModel->vertexPositionSB);
  calcDecalInfoDT.AddBufferSrv(parentModel->indexBuffer);
  calcDecalInfoDT.AddBufferUav(decalInfoSB);

  // create descriptor table for rendering indirectly decal into decal lookup-map
  renderDecalDT.AddBufferSrv(parentModel->vertexPositionSB);
  renderDecalDT.AddBufferSrv(parentModel->vertexNormalSB);
  renderDecalDT.AddBufferSrv(parentModel->vertexUvHandednessSB);
  renderDecalDT.AddBufferSrv(decalInfoSB);
  renderDecalDT.AddBufferUav(decalValiditySB);
  for(UINT i=0; i<numDecalLookupTextures; i++)
  {
    renderDecalDT.AddTextureUav(decalLookupTextures[i]);
  }

  // create descriptor table for removing decal
  removeDecalDT.AddBufferSrv(parentModel->vertexPositionSB);
  removeDecalDT.AddBufferSrv(parentModel->vertexUvHandednessSB);
  removeDecalDT.AddBufferSrv(parentModel->indexBuffer);
  removeDecalDT.AddBufferSrv(decalInfoSB);
  for(UINT i=0; i<numDecalLookupTextures; i++)
  {
    removeDecalDT.AddTextureSrv(decalLookupTextures[i]);
  }

  // create desciptor table for reading decal buffers
  decalBuffersDT.AddBufferSrv(decalInfoSB);
  decalBuffersDT.AddBufferSrv(decalValiditySB);
  for(UINT i=0; i<numDecalLookupTextures; i++)
  {
    decalBuffersDT.AddTextureSrv(decalLookupTextures[i]);
  }

  // create desciptor table for reading decal materials
  for(UINT i=0; i<numMaterials; i++)
  {
    Material *material = Demo::resourceManager->LoadMaterial(materialNames[i]);
    if(!material)
      return false;

    if((!material->textures[COLOR_TEX_ID]) || (!material->textures[NORMAL_TEX_ID]) || (!material->textures[SPECULAR_TEX_ID]))
    {
      return false;
    }

    decalMaterialsDT.AddTextureSrv(material->textures[COLOR_TEX_ID]);
    decalMaterialsDT.AddTextureSrv(material->textures[NORMAL_TEX_ID]);
    decalMaterialsDT.AddTextureSrv(material->textures[SPECULAR_TEX_ID]);
  }

  ClearSkinnedDecals();

  return true;
}

void SkinnedDecals::AddSkinnedDecal(const AddDecalInfo &decalInfo)
{
  if((decalConstData.decalIndex >= (MAX_NUM_SKINNED_DECALS - 1)) || (decalInfo.decalMaterialIndex >= numDecalMaterials))
    return;

  decalConstData.rayOrigin.Set(decalInfo.rayOrigin);
  decalConstData.rayDir.Set(decalInfo.rayDir);
  decalConstData.decalTangent.Set(decalInfo.decalTangent);
  float hitDistanceRange = decalInfo.maxHitDistance - decalInfo.minHitDistance;
  decalConstData.hitDistances.Set(decalInfo.minHitDistance, hitDistanceRange, 1.0f / hitDistanceRange, 0.0f);
  decalConstData.decalSizeX = decalInfo.decalSize.x;
  decalConstData.decalSizeY = decalInfo.decalSize.y;
  decalConstData.decalSizeZ = decalInfo.decalSize.z; 
  decalConstData.decalMaterialIndex = decalInfo.decalMaterialIndex;
  decalConstData.decalIndex++;

  addSkinnedDecal = true;
}

void SkinnedDecals::RemoveSkinnedDecal(const RemoveDecalInfo &decalInfo)
{
  if(decalConstData.decalIndex == 0)
    return;

  decalConstData.rayOrigin.Set(decalInfo.rayOrigin);
  decalConstData.rayDir.Set(decalInfo.rayDir);
  float hitDistanceRange = decalInfo.maxHitDistance - decalInfo.minHitDistance;
  decalConstData.hitDistances.Set(decalInfo.minHitDistance, hitDistanceRange, 1.0f / hitDistanceRange, 0.0f);

  removeSkinnedDecal = true;
}

void SkinnedDecals::UpdateBuffers()
{
  // update constant buffer
  decalCB->Update(&decalConstData, 1);
}

void SkinnedDecals::PerformClearSkinnedDecals()
{
  DX12_CmdList *cmdList = Demo::renderer->GetCmdList(DEFAULT_CMD_LIST_ID);

  SCOPED_GPU_MARKER(cmdList, DECAL_GPU_CMD_ORDER, "Clear skinned decals");

  {
    ResourceBarrier barriers[MAX_NUM_SUBMODELS + 1];
    UINT numBarriers = 0;
    for(UINT i=0; i<numDecalLookupTextures; i++)
    {
      barriers[numBarriers].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
      barriers[numBarriers].transition.resource = decalLookupTextures[i];
      barriers[numBarriers].transition.newResourceState = UNORDERED_ACCESS_RESOURCE_STATE;
      numBarriers++; 
    }
    barriers[numBarriers].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
    barriers[numBarriers].transition.resource = decalValiditySB;
    barriers[numBarriers].transition.newResourceState = UNORDERED_ACCESS_RESOURCE_STATE;
    numBarriers++;
 
    ResourceBarrierCmd cmd;
    cmd.barriers = barriers;
    cmd.numBarriers = numBarriers;
    cmdList->AddGpuCmd(cmd, DECAL_GPU_CMD_ORDER);
  }

  // clear decal lookup textures
  for(UINT i=0; i<numDecalLookupTextures; i++)
  {
    UINT width = decalLookupTextures[i]->GetTextureDesc().width;
    UINT height = decalLookupTextures[i]->GetTextureDesc().height;

    RootParam rootParams[3];
    rootParams[0].rootParamType = CONST_ROOT_PARAM;
    rootParams[0].rootConst.constData[0] = i;
    rootParams[0].rootConst.numConsts = 1;
    rootParams[1].rootParamType = CBV_ROOT_PARAM;
    rootParams[1].bufferLocation = decalCB->GetGpuVirtualAddress();
    rootParams[2].rootParamType = DESC_TABLE_ROOT_PARAM;
    rootParams[2].baseGpuDescHandle = clearDecalsDT.GetBaseDescHandle().gpuDescHandle;

    ComputeCmd cmd;
    cmd.numThreadGroupsX = (width + CLEAR_DECALLOOKUPMAP_THREAD_GROUP_SIZE - 1) / CLEAR_DECALLOOKUPMAP_THREAD_GROUP_SIZE;
    cmd.numThreadGroupsY = (height + CLEAR_DECALLOOKUPMAP_THREAD_GROUP_SIZE - 1) / CLEAR_DECALLOOKUPMAP_THREAD_GROUP_SIZE;
    cmd.numThreadGroupsZ = 1;
    cmd.pipelineState = clearDecalLookupMapPS;
    cmd.rootParams = rootParams;
    cmd.numRootParams = _countof(rootParams);
    cmdList->AddGpuCmd(cmd, DECAL_GPU_CMD_ORDER);  
  }

  {
    // clear decal validity buffer
    RootParam rootParams[1];
    rootParams[0].rootParamType = UAV_ROOT_PARAM;
    rootParams[0].bufferLocation = decalValiditySB->GetGpuVirtualAddress();

    ComputeCmd cmd;
    cmd.numThreadGroupsX = 1;
    cmd.numThreadGroupsY = 1;
    cmd.numThreadGroupsZ = 1;
    cmd.pipelineState = clearDecalValidityBufferPS;
    cmd.rootParams = rootParams;
    cmd.numRootParams = _countof(rootParams);
    cmdList->AddGpuCmd(cmd, DECAL_GPU_CMD_ORDER);
  }

  {
    ResourceBarrier barriers[MAX_NUM_SUBMODELS + 1];
    UINT numBarriers = 0;
    for(UINT i=0; i<numDecalLookupTextures; i++)
    {
      barriers[numBarriers].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
      barriers[numBarriers].transition.resource = decalLookupTextures[i];
      barriers[numBarriers].transition.newResourceState = PIXEL_SHADER_RESOURCE_RESOURCE_STATE;
      numBarriers++;
    }
    barriers[numBarriers].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
    barriers[numBarriers].transition.resource = decalValiditySB;
    barriers[numBarriers].transition.newResourceState = PIXEL_SHADER_RESOURCE_RESOURCE_STATE;
    numBarriers++;
 
    ResourceBarrierCmd cmd;
    cmd.barriers = barriers;
    cmd.numBarriers = numBarriers;
    cmdList->AddGpuCmd(cmd, DECAL_GPU_CMD_ORDER);
  }
}

void SkinnedDecals::PerformAddSkinnedDecal()
{
  DX12_CmdList *cmdList = Demo::renderer->GetCmdList(DEFAULT_CMD_LIST_ID);

  SCOPED_GPU_MARKER(cmdList, DECAL_GPU_CMD_ORDER, "Add skinned decal");

  {
    ResourceBarrier barriers[2];
    barriers[0].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
    barriers[0].transition.resource = parentModel->indexBuffer;
    barriers[0].transition.newResourceState = NON_PIXEL_SHADER_RESOURCE_RESOURCE_STATE;
    barriers[1].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
    barriers[1].transition.resource = decalInfoSB;
    barriers[1].transition.newResourceState = UNORDERED_ACCESS_RESOURCE_STATE;

    ResourceBarrierCmd cmd;
    cmd.barriers = barriers;
    cmd.numBarriers = _countof(barriers);
    cmdList->AddGpuCmd(cmd, DECAL_GPU_CMD_ORDER);
  }

  {
    // clear decal info
    RootParam rootParams[1];
    rootParams[0].rootParamType = DESC_TABLE_ROOT_PARAM;
    rootParams[0].baseGpuDescHandle = clearDecalInfoDT.GetBaseDescHandle().gpuDescHandle;

    ComputeCmd cmd;
    cmd.numThreadGroupsX = 1;
    cmd.numThreadGroupsY = 1;
    cmd.numThreadGroupsZ = 1;
    cmd.pipelineState = clearDecalInfoPS;
    cmd.rootParams = rootParams;
    cmd.numRootParams = _countof(rootParams);
    cmdList->AddGpuCmd(cmd, DECAL_GPU_CMD_ORDER);
  }

  {
    ResourceBarrier barriers[1];
    barriers[0].barrierType = UAV_RESOURCE_BARRIER_TYPE;
    barriers[0].uav.resource = decalInfoSB;

    ResourceBarrierCmd cmd;
    cmd.barriers = barriers;
    cmd.numBarriers = _countof(barriers);
    cmdList->AddGpuCmd(cmd, DECAL_GPU_CMD_ORDER);
  }

  {
    // find closest intersection
    UINT decalMeshIndex = 0;
    for(UINT i=0; i<parentModel->numSubModels; i++)
    {
      if(parentModel->subModels[i].material->receiveDecals)
      {
        UINT numTriangles = parentModel->subModels[i].numIndices / 3;

        RootParam rootParams[3];
        rootParams[0].rootParamType = CONST_ROOT_PARAM;
        rootParams[0].rootConst.constData[0] = decalMeshIndex++;
        rootParams[0].rootConst.numConsts = 1;
        rootParams[1].rootParamType = CBV_ROOT_PARAM;
        rootParams[1].bufferLocation = decalCB->GetGpuVirtualAddress();
        rootParams[2].rootParamType = DESC_TABLE_ROOT_PARAM;
        rootParams[2].baseGpuDescHandle = calcDecalInfoDT.GetBaseDescHandle().gpuDescHandle;

        ComputeCmd cmd;
        cmd.numThreadGroupsX = (numTriangles + RAYCAST_THREAD_GROUP_SIZE - 1) / RAYCAST_THREAD_GROUP_SIZE;
        cmd.numThreadGroupsY = 1;
        cmd.numThreadGroupsZ = 1;
        cmd.pipelineState = raycastPS;
        cmd.rootParams = rootParams;
        cmd.numRootParams = _countof(rootParams);
        cmdList->AddGpuCmd(cmd, DECAL_GPU_CMD_ORDER);
      }
    }
  }

  {
    ResourceBarrier barriers[1];
    barriers[0].barrierType = UAV_RESOURCE_BARRIER_TYPE;
    barriers[0].uav.resource = decalInfoSB;

    ResourceBarrierCmd cmd;
    cmd.barriers = barriers;
    cmd.numBarriers = _countof(barriers);
    cmdList->AddGpuCmd(cmd, DECAL_GPU_CMD_ORDER);
  }

  {
    // calculate decal info
    RootParam rootParams[2];
    rootParams[0].rootParamType = CBV_ROOT_PARAM;
    rootParams[0].bufferLocation = decalCB->GetGpuVirtualAddress();
    rootParams[1].rootParamType = DESC_TABLE_ROOT_PARAM;
    rootParams[1].baseGpuDescHandle = calcDecalInfoDT.GetBaseDescHandle().gpuDescHandle;

    ComputeCmd cmd;
    cmd.numThreadGroupsX = 1;
    cmd.numThreadGroupsY = 1;
    cmd.numThreadGroupsZ = 1;
    cmd.pipelineState = calcDecalInfoPS;
    cmd.rootParams = rootParams;
    cmd.numRootParams = _countof(rootParams);
    cmdList->AddGpuCmd(cmd, DECAL_GPU_CMD_ORDER);
  }

  {
    ResourceBarrier barriers[MAX_NUM_SUBMODELS + 3];
    UINT numBarriers = 0;
    for(UINT i=0; i<numDecalLookupTextures; i++)
    {
      barriers[numBarriers].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
      barriers[numBarriers].transition.resource = decalLookupTextures[i];
      barriers[numBarriers].transition.newResourceState = UNORDERED_ACCESS_RESOURCE_STATE;
      numBarriers++;
    }
    barriers[numBarriers].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
    barriers[numBarriers].transition.resource = decalValiditySB;
    barriers[numBarriers].transition.newResourceState = UNORDERED_ACCESS_RESOURCE_STATE;
    numBarriers++;
    barriers[numBarriers].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
    barriers[numBarriers].transition.resource = parentModel->indexBuffer;
    barriers[numBarriers].transition.newResourceState = INDEX_BUFFER_RESOURCE_STATE;
    numBarriers++;
    barriers[numBarriers].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
    barriers[numBarriers].transition.resource = decalInfoSB;
    barriers[numBarriers].transition.newResourceState = NON_PIXEL_SHADER_RESOURCE_RESOURCE_STATE |
      PIXEL_SHADER_RESOURCE_RESOURCE_STATE | INDIRECT_ARGUMENT_RESOURCE_STATE;
    numBarriers++;
 
    ResourceBarrierCmd cmd;
    cmd.barriers = barriers;
    cmd.numBarriers = numBarriers;
    cmdList->AddGpuCmd(cmd, DECAL_GPU_CMD_ORDER);
  }

  {
    // render decal indirectly into decal lookup map
    CpuDescHandle rtvCpuDescHandles[] = 
    {
      decalLookupMapRT->GetRtv().cpuDescHandle
    };

    RootParam rootParams[1];
    rootParams[0].rootParamType = DESC_TABLE_ROOT_PARAM;
    rootParams[0].baseGpuDescHandle = renderDecalDT.GetBaseDescHandle().gpuDescHandle;

    IndirectDrawCmd cmd;
    cmd.rtvCpuDescHandles = rtvCpuDescHandles;
    cmd.numRenderTargets = _countof(rtvCpuDescHandles);
    cmd.viewportSet = decalLookupMapVPS;
    cmd.scissorRectSet = decalLookupMapSRS;
    cmd.indexBuffer = parentModel->indexBuffer;
    cmd.primitiveTopology = TRIANGLELIST_PRIMITIVE_TOPOLOGY;
    cmd.cmdSignature = renderDecalCS;
    cmd.argBuffer = decalInfoSB;
    cmd.maxCmdCount = 1;
    cmd.pipelineState = renderDecalPS;
    cmd.rootParams = rootParams;
    cmd.numRootParams = _countof(rootParams);
    cmdList->AddGpuCmd(cmd, DECAL_GPU_CMD_ORDER);
  }

  {
    ResourceBarrier barriers[MAX_NUM_SUBMODELS + 1];
    UINT numBarriers = 0;
    for(UINT i=0; i<numDecalLookupTextures; i++)
    {
      barriers[numBarriers].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
      barriers[numBarriers].transition.resource = decalLookupTextures[i];
      barriers[numBarriers].transition.newResourceState = PIXEL_SHADER_RESOURCE_RESOURCE_STATE;
      numBarriers++;
    }
    barriers[numBarriers].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
    barriers[numBarriers].transition.resource = decalValiditySB;
    barriers[numBarriers].transition.newResourceState = PIXEL_SHADER_RESOURCE_RESOURCE_STATE;
    numBarriers++;
 
    ResourceBarrierCmd cmd;
    cmd.barriers = barriers;
    cmd.numBarriers = numBarriers;
    cmdList->AddGpuCmd(cmd, DECAL_GPU_CMD_ORDER);
  }
}

void SkinnedDecals::PerformRemoveSkinnedDecal()
{
  DX12_CmdList *cmdList = Demo::renderer->GetCmdList(DEFAULT_CMD_LIST_ID);

  SCOPED_GPU_MARKER(cmdList, DECAL_GPU_CMD_ORDER, "Remove skinned decal");

  {
    ResourceBarrier barriers[MAX_NUM_SUBMODELS + 3];
    UINT numBarriers = 0;
    barriers[numBarriers].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
    barriers[numBarriers].transition.resource = parentModel->indexBuffer;
    barriers[numBarriers].transition.newResourceState = NON_PIXEL_SHADER_RESOURCE_RESOURCE_STATE;
    numBarriers++;
    barriers[numBarriers].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
    barriers[numBarriers].transition.resource = decalInfoSB;
    barriers[numBarriers].transition.newResourceState = UNORDERED_ACCESS_RESOURCE_STATE;
    numBarriers++;
    barriers[numBarriers].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
    barriers[numBarriers].transition.resource = decalValiditySB;
    barriers[numBarriers].transition.newResourceState = UNORDERED_ACCESS_RESOURCE_STATE;
    numBarriers++;
    for(UINT i=0; i<numDecalLookupTextures; i++)
    {
      barriers[numBarriers].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
      barriers[numBarriers].transition.resource = decalLookupTextures[i];
      barriers[numBarriers].transition.newResourceState = NON_PIXEL_SHADER_RESOURCE_RESOURCE_STATE;
      numBarriers++; 
    }

    ResourceBarrierCmd cmd;
    cmd.barriers = barriers;
    cmd.numBarriers = numBarriers;
    cmdList->AddGpuCmd(cmd, DECAL_GPU_CMD_ORDER);
  }

  {
    // clear decal info
    RootParam rootParams[1];
    rootParams[0].rootParamType = DESC_TABLE_ROOT_PARAM;
    rootParams[0].baseGpuDescHandle = clearDecalInfoDT.GetBaseDescHandle().gpuDescHandle;

    ComputeCmd cmd;
    cmd.numThreadGroupsX = 1;
    cmd.numThreadGroupsY = 1;
    cmd.numThreadGroupsZ = 1;
    cmd.pipelineState = clearDecalInfoPS;
    cmd.rootParams = rootParams;
    cmd.numRootParams = _countof(rootParams);
    cmdList->AddGpuCmd(cmd, DECAL_GPU_CMD_ORDER);
  }

  {
    ResourceBarrier barriers[1];
    barriers[0].barrierType = UAV_RESOURCE_BARRIER_TYPE;
    barriers[0].uav.resource = decalInfoSB;

    ResourceBarrierCmd cmd;
    cmd.barriers = barriers;
    cmd.numBarriers = _countof(barriers);
    cmdList->AddGpuCmd(cmd, DECAL_GPU_CMD_ORDER);
  }

  {
    // find closest intersection
    UINT decalMeshIndex = 0;
    for(UINT i=0; i<parentModel->numSubModels; i++)
    {
      if(parentModel->subModels[i].material->receiveDecals)
      {
        UINT numTriangles = parentModel->subModels[i].numIndices / 3;

        RootParam rootParams[3];
        rootParams[0].rootParamType = CONST_ROOT_PARAM;
        rootParams[0].rootConst.constData[0] = decalMeshIndex++;
        rootParams[0].rootConst.numConsts = 1;
        rootParams[1].rootParamType = CBV_ROOT_PARAM;
        rootParams[1].bufferLocation = decalCB->GetGpuVirtualAddress();
        rootParams[2].rootParamType = DESC_TABLE_ROOT_PARAM;
        rootParams[2].baseGpuDescHandle = calcDecalInfoDT.GetBaseDescHandle().gpuDescHandle;

        ComputeCmd cmd;
        cmd.numThreadGroupsX = (numTriangles + RAYCAST_THREAD_GROUP_SIZE - 1) / RAYCAST_THREAD_GROUP_SIZE;
        cmd.numThreadGroupsY = 1;
        cmd.numThreadGroupsZ = 1;
        cmd.pipelineState = raycastPS;
        cmd.rootParams = rootParams;
        cmd.numRootParams = _countof(rootParams);
        cmdList->AddGpuCmd(cmd, DECAL_GPU_CMD_ORDER);
      }
    }
  }

  {
    ResourceBarrier barriers[1];
    barriers[0].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
    barriers[0].transition.resource = decalInfoSB;
    barriers[0].transition.newResourceState = NON_PIXEL_SHADER_RESOURCE_RESOURCE_STATE;

    ResourceBarrierCmd cmd;
    cmd.barriers = barriers;
    cmd.numBarriers = _countof(barriers);
    cmdList->AddGpuCmd(cmd, DECAL_GPU_CMD_ORDER);
  }

  {
    // remove decal
    RootParam rootParams[3];
    rootParams[0].rootParamType = CBV_ROOT_PARAM;
    rootParams[0].bufferLocation = decalCB->GetGpuVirtualAddress();
    rootParams[1].rootParamType = DESC_TABLE_ROOT_PARAM;
    rootParams[1].baseGpuDescHandle = removeDecalDT.GetBaseDescHandle().gpuDescHandle;
    rootParams[2].rootParamType = UAV_ROOT_PARAM;
    rootParams[2].bufferLocation = decalValiditySB->GetGpuVirtualAddress();

    ComputeCmd cmd;
    cmd.numThreadGroupsX = 1;
    cmd.numThreadGroupsY = 1;
    cmd.numThreadGroupsZ = 1;
    cmd.pipelineState = removeDecalPS;
    cmd.rootParams = rootParams;
    cmd.numRootParams = _countof(rootParams);
    cmdList->AddGpuCmd(cmd, DECAL_GPU_CMD_ORDER);
  }

  {
    ResourceBarrier barriers[MAX_NUM_SUBMODELS + 3];
    UINT numBarriers = 0;
    barriers[numBarriers].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
    barriers[numBarriers].transition.resource = parentModel->indexBuffer;
    barriers[numBarriers].transition.newResourceState = INDEX_BUFFER_RESOURCE_STATE;
    numBarriers++;
    barriers[numBarriers].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
    barriers[numBarriers].transition.resource = decalInfoSB;
    barriers[numBarriers].transition.newResourceState = PIXEL_SHADER_RESOURCE_RESOURCE_STATE;
    numBarriers++;
    barriers[numBarriers].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
    barriers[numBarriers].transition.resource = decalValiditySB;
    barriers[numBarriers].transition.newResourceState = PIXEL_SHADER_RESOURCE_RESOURCE_STATE;
    numBarriers++;
    for(UINT i=0; i<numDecalLookupTextures; i++)
    {
      barriers[numBarriers].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
      barriers[numBarriers].transition.resource = decalLookupTextures[i];
      barriers[numBarriers].transition.newResourceState = PIXEL_SHADER_RESOURCE_RESOURCE_STATE;
      numBarriers++;
    }

    ResourceBarrierCmd cmd;
    cmd.barriers = barriers;
    cmd.numBarriers = numBarriers;
    cmdList->AddGpuCmd(cmd, DECAL_GPU_CMD_ORDER);
  }
}

void SkinnedDecals::Render()
{
  UpdateBuffers();

  if(clearSkinnedDecals)
  {
    PerformClearSkinnedDecals();
    clearSkinnedDecals = false;
  }

  if(addSkinnedDecal || removeSkinnedDecal)
  {
    if(addSkinnedDecal)
    {
      PerformAddSkinnedDecal();
      addSkinnedDecal = false;
    }

    if(removeSkinnedDecal)
    {
      PerformRemoveSkinnedDecal();
      removeSkinnedDecal = false;
    }
  }
}
