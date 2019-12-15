#include <stdafx.h>
#include <Demo.h>
#include <Shading.h>
#include <DemoModel.h>

void DemoModel::Release()
{
  SAFE_DELETE(skinnedDecals);
  SAFE_DELETE_ARRAY(subModels);
  if(animFrames)
  {
     for(UINT i=0; i<numAnimFrames; i++)
       SAFE_DELETE_ARRAY(animFrames[i].joints);
     SAFE_DELETE_ARRAY(animFrames);
  }
  SAFE_DELETE_ARRAY(interpAnimFrame.joints);
}

bool DemoModel::Load(const char *filename)
{
  // cache pointer to Shading post-processor
  shadingPP = (Shading*)Demo::renderer->GetPostProcessor("Shading");
  if(!shadingPP)
    return false;

  // load ".model" file
  char filePath[DEMO_MAX_FILEPATH];
  if(!Demo::fileManager->GetFilePath(filename, filePath))
    return false;
  FILE *file;
  fopen_s(&file, filePath, "rb");
  if(!file)
    return false;

  // check idString
  char idString[5];
  memset(idString, 0, 5);
  fread(idString, sizeof(char), 4, file);
  if(strcmp(idString, "DMDL") != 0)
  {
    fclose(file);
    return false;
  }

  // check version
  UINT version;
  fread(&version, sizeof(UINT), 1, file);
  if(version != CURRENT_DEMO_MODEL_VERSION)
  {
    fclose(file);
    return false;
  }

  // load number of sub-models
  fread(&numSubModels, sizeof(UINT), 1, file);
  assert(numSubModels <= MAX_NUM_SUBMODELS);

  // load sub-models
	subModels = new DemoSubModel[numSubModels];
	if(!subModels)
	{
    fclose(file);
    return false;
  }
  for(UINT i=0; i<numSubModels; i++)
  {
    DemoSubModel &subModel = subModels[i];
    char materialName[256];
    fread(materialName, sizeof(char), 256, file);
    subModel.material = Demo::resourceManager->LoadMaterial(materialName);
    if(!subModel.material)
    {
      fclose(file);
      return false;
    }

    RootSignatureDesc rootSignatureDesc;
    // camera constant buffer
    rootSignatureDesc.rootParamDescs[0].rootParamType = CBV_ROOT_PARAM;
    rootSignatureDesc.rootParamDescs[0].rootDesc.shaderReg = 0;
    rootSignatureDesc.rootParamDescs[0].rootDesc.regSpace = 0;
    rootSignatureDesc.rootParamDescs[0].shaderVisibility = ALL_SHADER_VIS;
    // lighting info constant buffer
    rootSignatureDesc.rootParamDescs[1].rootParamType = CBV_ROOT_PARAM;
    rootSignatureDesc.rootParamDescs[1].rootDesc.shaderReg = 1;
    rootSignatureDesc.rootParamDescs[1].rootDesc.regSpace = 0;
    rootSignatureDesc.rootParamDescs[1].shaderVisibility = PS_SHADER_VIS;
    // model constant buffer
    rootSignatureDesc.rootParamDescs[2].rootParamType = CBV_ROOT_PARAM;
    rootSignatureDesc.rootParamDescs[2].rootDesc.shaderReg = 2;
    rootSignatureDesc.rootParamDescs[2].rootDesc.regSpace = 0;
    rootSignatureDesc.rootParamDescs[2].shaderVisibility = PS_SHADER_VIS;
    // vertex attributes SRVs
    rootSignatureDesc.rootParamDescs[3].rootParamType = DESC_TABLE_ROOT_PARAM;
    rootSignatureDesc.rootParamDescs[3].rootDescTableDesc.numRanges = 1;
    rootSignatureDesc.rootParamDescs[3].rootDescTableDesc.ranges[0].rangeType = SRV_ROOT_DESC_TABLE_RANGE;
    rootSignatureDesc.rootParamDescs[3].rootDescTableDesc.ranges[0].numDescs = 4;
    rootSignatureDesc.rootParamDescs[3].rootDescTableDesc.ranges[0].baseShaderReg = 0;
    rootSignatureDesc.rootParamDescs[3].rootDescTableDesc.ranges[0].regSpace = 0;
    rootSignatureDesc.rootParamDescs[3].shaderVisibility = VS_SHADER_VIS;
    // material textures SRVs
    rootSignatureDesc.rootParamDescs[4].rootParamType = DESC_TABLE_ROOT_PARAM;
    rootSignatureDesc.rootParamDescs[4].rootDescTableDesc.numRanges = 1;
    rootSignatureDesc.rootParamDescs[4].rootDescTableDesc.ranges[0].rangeType = SRV_ROOT_DESC_TABLE_RANGE;
    rootSignatureDesc.rootParamDescs[4].rootDescTableDesc.ranges[0].numDescs = 3;
    rootSignatureDesc.rootParamDescs[4].rootDescTableDesc.ranges[0].baseShaderReg = 0;
    rootSignatureDesc.rootParamDescs[4].rootDescTableDesc.ranges[0].regSpace = 1;
    rootSignatureDesc.rootParamDescs[4].shaderVisibility = PS_SHADER_VIS;
    rootSignatureDesc.numRootParamDescs = 5;
    // texture sampler
    rootSignatureDesc.samplerDescs[0].filter = MIN_MAG_MIP_LINEAR_FILTER;
    rootSignatureDesc.samplerDescs[0].adressU = WRAP_TEX_ADDRESS;
    rootSignatureDesc.samplerDescs[0].adressV = WRAP_TEX_ADDRESS;
    rootSignatureDesc.samplerDescs[0].adressW = WRAP_TEX_ADDRESS;
    rootSignatureDesc.samplerDescs[0].shaderReg = 0;
    rootSignatureDesc.samplerDescs[0].regSpace = 1;
    rootSignatureDesc.samplerDescs[0].shaderVisibility = PS_SHADER_VIS;
    rootSignatureDesc.numSamplerDescs = 1;
    DX12_RootSignature *rootSignature = Demo::renderer->CreateRootSignature(rootSignatureDesc, "Base No Decals");
    if(!rootSignature)
    {
      fclose(file);
      return false;
    }

    PipelineStateDesc pipelineStateDesc(GRAPHICS_PIPELINE_STATE);
    pipelineStateDesc.rootSignature = rootSignature;
    pipelineStateDesc.graphics.shader = subModel.material->shader;
    pipelineStateDesc.graphics.numRenderTargets = 1;
    pipelineStateDesc.graphics.rtvFormats[0] = Demo::renderer->GetRenderTarget(ACCUM_BUFFER_RT_ID)->GetTexture()->GetTextureDesc().format;
    pipelineStateDesc.graphics.dsvFormat = Demo::renderer->GetDepthStencilTarget(MAIN_DEPTH_DST_ID)->GetTexture()->GetTextureDesc().format;
    pipelineStateDesc.graphics.blendDesc = subModel.material->blendDesc;
    pipelineStateDesc.graphics.rasterizerDesc = subModel.material->rasterDesc;
    pipelineStateDesc.graphics.depthStencilDesc = subModel.material->depthStencilDesc;
    subModel.baseNoDecalsPS = Demo::renderer->CreatePipelineState(pipelineStateDesc, "Base No Decals");
    if(!subModel.baseNoDecalsPS)
    {
      fclose(file);
      return false;
    }

    if(subModel.material->receiveDecals)
    {
      // decal mesh info root constant
      rootSignatureDesc.rootParamDescs[5].rootParamType = CONST_ROOT_PARAM;
      rootSignatureDesc.rootParamDescs[5].rootConstDesc.numConsts = 1;
      rootSignatureDesc.rootParamDescs[5].rootConstDesc.shaderReg = 3;
      rootSignatureDesc.rootParamDescs[5].rootConstDesc.regSpace = 0;
      rootSignatureDesc.rootParamDescs[5].shaderVisibility = PS_SHADER_VIS;
      // decal constant buffer
      rootSignatureDesc.rootParamDescs[6].rootParamType = CBV_ROOT_PARAM;
      rootSignatureDesc.rootParamDescs[6].rootDesc.shaderReg = 4;
      rootSignatureDesc.rootParamDescs[6].rootDesc.regSpace = 0;
      rootSignatureDesc.rootParamDescs[6].shaderVisibility = PS_SHADER_VIS;
      // decal info buffer/ validity buffer/ lookup textures SRVs
      rootSignatureDesc.rootParamDescs[7].rootParamType = DESC_TABLE_ROOT_PARAM;
      rootSignatureDesc.rootParamDescs[7].rootDescTableDesc.numRanges = 1;
      rootSignatureDesc.rootParamDescs[7].rootDescTableDesc.ranges[0].rangeType = SRV_ROOT_DESC_TABLE_RANGE;
      rootSignatureDesc.rootParamDescs[7].rootDescTableDesc.ranges[0].numDescs = 2 + MAX_NUM_SUBMODELS;
      rootSignatureDesc.rootParamDescs[7].rootDescTableDesc.ranges[0].baseShaderReg = 0;
      rootSignatureDesc.rootParamDescs[7].rootDescTableDesc.ranges[0].regSpace = 2;
      rootSignatureDesc.rootParamDescs[7].shaderVisibility = PS_SHADER_VIS;
      // decal material SRVs
      rootSignatureDesc.rootParamDescs[8].rootParamType = DESC_TABLE_ROOT_PARAM;
      rootSignatureDesc.rootParamDescs[8].rootDescTableDesc.numRanges = 1;
      rootSignatureDesc.rootParamDescs[8].rootDescTableDesc.ranges[0].rangeType = SRV_ROOT_DESC_TABLE_RANGE;
      rootSignatureDesc.rootParamDescs[8].rootDescTableDesc.ranges[0].numDescs = MAX_NUM_DECAL_TEXTURES;
      rootSignatureDesc.rootParamDescs[8].rootDescTableDesc.ranges[0].baseShaderReg = 0;
      rootSignatureDesc.rootParamDescs[8].rootDescTableDesc.ranges[0].regSpace = 3;
      rootSignatureDesc.rootParamDescs[8].shaderVisibility = PS_SHADER_VIS;
      rootSignatureDesc.numRootParamDescs = 9;
      // decal lookup texture sampler
      rootSignatureDesc.samplerDescs[1].filter = MIN_MAG_LINEAR_MIP_POINT_FILTER;
      rootSignatureDesc.samplerDescs[1].adressU = WRAP_TEX_ADDRESS;
      rootSignatureDesc.samplerDescs[1].adressV = WRAP_TEX_ADDRESS;
      rootSignatureDesc.samplerDescs[1].adressW = WRAP_TEX_ADDRESS;
      rootSignatureDesc.samplerDescs[1].shaderReg = 1;
      rootSignatureDesc.samplerDescs[1].regSpace = 1;
      rootSignatureDesc.samplerDescs[1].shaderVisibility = PS_SHADER_VIS;
      rootSignatureDesc.numSamplerDescs = 2;
      rootSignature = Demo::renderer->CreateRootSignature(rootSignatureDesc, "Base With Decals");
      if(!rootSignature)
      {
        fclose(file);
        return false;
      }

      UINT permutationMask = subModel.material->shader->GetPermutationMask() | 0x2; // 0x2 = RECEIVE_DECALS
      if(Demo::renderer->GetDeviceFeatureSupport().typedUavLoads)
        permutationMask |= 0x4; // 0x4 = TYPED_UAV_LOADS
      DX12_Shader *shader = Demo::resourceManager->LoadShader(subModel.material->shader->GetName(), permutationMask);
      if(!shader)
      {
        fclose(file);
        return false;
      }

      pipelineStateDesc.rootSignature = rootSignature;
      pipelineStateDesc.graphics.shader = shader;
      subModel.baseWithDecalsPS = Demo::renderer->CreatePipelineState(pipelineStateDesc, "Base With Decals");
      if(!subModel.baseWithDecalsPS)
      {
        fclose(file);
        return false;
      }
    }

    fread(&subModel.firstIndex, sizeof(UINT), 1, file);
    fread(&subModel.numIndices, sizeof(UINT), 1, file);

    if((!subModel.material->textures[COLOR_TEX_ID]) || (!subModel.material->textures[NORMAL_TEX_ID]) || (!subModel.material->textures[SPECULAR_TEX_ID]))
    {
      fclose(file);
      return false;
    }

    subModel.materialDT.AddTextureSrv(subModel.material->textures[COLOR_TEX_ID]);
    subModel.materialDT.AddTextureSrv(subModel.material->textures[NORMAL_TEX_ID]);
    subModel.materialDT.AddTextureSrv(subModel.material->textures[SPECULAR_TEX_ID]);
  }

  // load number of vertices
  fread(&modelConstData.numVertices, sizeof(UINT), 1, file);

  {
    // load weight indices
    WeightIndex *weightIndices = new WeightIndex[modelConstData.numVertices];
    if(!weightIndices)
    {
      fclose(file);
      return false;
    }
    fread(weightIndices, sizeof(WeightIndex), modelConstData.numVertices, file);

    // create structured buffer for weight indices
    BufferDesc desc;
    desc.bufferType = STRUCTURED_BUFFER;
    desc.elementSize = sizeof(WeightIndex);
    desc.numElements = modelConstData.numVertices;
    desc.flags = CPU_WRITE_BUFFER_FLAG;
    weightIndexSB = Demo::renderer->CreateBuffer(desc, "WeightIndex");
    if(!weightIndexSB)  
    {
      SAFE_DELETE_ARRAY(weightIndices);
      fclose(file);
      return false;
    }
    weightIndexSB->Update(weightIndices, modelConstData.numVertices);
    SAFE_DELETE_ARRAY(weightIndices);
  }
  
  {
    // load texCoords and handedness
    Vector3 *vertexUvHandedness = new Vector3[modelConstData.numVertices];
    if(!vertexUvHandedness)
    {
      fclose(file);
      return false;
    }
    fread(vertexUvHandedness, sizeof(Vector3), modelConstData.numVertices, file);

    // create structured buffer for vertex texCoords and handedness
    BufferDesc desc;
    desc.bufferType = STRUCTURED_BUFFER;
    desc.elementSize = sizeof(Vector3);
    desc.numElements = modelConstData.numVertices;
    desc.flags = CPU_WRITE_BUFFER_FLAG;
    vertexUvHandednessSB = Demo::renderer->CreateBuffer(desc, "UvHandedness");
    if(!vertexUvHandednessSB)  
    {
      SAFE_DELETE_ARRAY(vertexUvHandedness);
      fclose(file);
      return false;
    }
    vertexUvHandednessSB->Update(vertexUvHandedness, modelConstData.numVertices);
    SAFE_DELETE_ARRAY(vertexUvHandedness);
  }
  
  // load number of weights
  UINT numWeights;
  fread(&numWeights, sizeof(UINT), 1, file);

  {
    // load weights
    Weight *weights = new Weight[numWeights];
    if(!weights)
    {
      fclose(file);
      return false;
    }
    fread(weights, sizeof(Weight), numWeights, file);

    // create structured buffer for weights
    BufferDesc desc;
    desc.bufferType = STRUCTURED_BUFFER;
    desc.elementSize = sizeof(Weight);
    desc.numElements = numWeights;
    desc.flags = CPU_WRITE_BUFFER_FLAG;
    weightSB = Demo::renderer->CreateBuffer(desc, "Weight");
    if(!weightSB)  
    {
      SAFE_DELETE_ARRAY(weights);
      fclose(file);
      return false;
    }
    weightSB->Update(weights, numWeights);
    SAFE_DELETE_ARRAY(weights);
  }
  
  // load number of indices
  UINT numIndices;
  fread(&numIndices, sizeof(UINT), 1, file);

  {
    // load indices
    UINT *indices = new UINT[numIndices];
    if(!indices)
    {
      fclose(file);
      return false;
    }
    fread(indices, sizeof(UINT), numIndices, file);

    // create index buffer
    BufferDesc desc;
    desc.bufferType = INDEX_BUFFER;
    desc.elementSize = sizeof(unsigned int);
    desc.numElements = numIndices;
    desc.elementFormat = R32UI_RENDER_FORMAT;
    desc.flags = CPU_WRITE_BUFFER_FLAG;
    indexBuffer = Demo::renderer->CreateBuffer(desc, "DemoModel");
    if(!indexBuffer)
    {
      SAFE_DELETE_ARRAY(indices);
      fclose(file);
      return false;
    }
    indexBuffer->Update(indices, numIndices);
    SAFE_DELETE_ARRAY(indices);
  }

  // load animation frame-rate and init animation timer
  UINT animFrameRate;
  fread(&animFrameRate, sizeof(UINT), 1, file);
  animTimer.SetInterval(static_cast<double>(animFrameRate));

  // load number of joints
  fread(&numJoints, sizeof(UINT), 1, file);

  {
    // create structured buffer for interpolated joints
    BufferDesc desc;
    desc.bufferType = STRUCTURED_BUFFER;
    desc.elementSize = sizeof(Joint);
    desc.numElements = numJoints;
    desc.flags = CPU_WRITE_BUFFER_FLAG | DYNAMIC_BUFFER_FLAG;
    jointSB = Demo::renderer->CreateBuffer(desc, "Joint");
    if(!jointSB) 
    {
      fclose(file);
      return false;
    } 
  }
  
  // load number of animation frames
  fread(&numAnimFrames, sizeof(UINT), 1, file);

  // load animation frames
	animFrames = new AnimFrame[numAnimFrames];
	if(!animFrames)
	{
    fclose(file);
    return false;
  }
  for(UINT i=0; i<numAnimFrames; i++)
  {
    animFrames[i].joints = new Joint[numJoints];
  }
  for(UINT i=0; i<numAnimFrames; i++)
  {
    if(!animFrames[i].joints)
    {
      fclose(file);
      return false;
    }
    fread(animFrames[i].joints, sizeof(Joint), numJoints, file);
    fread(&animFrames[i].bounds, sizeof(Aabb), 1, file);
  }

  fclose(file);  

  // create current animation frame
  interpAnimFrame.joints = new Joint[numJoints];
  if(!interpAnimFrame.joints)
    return false;

  // create structured buffers for vertex positions/ normals/ tangents
  {
    BufferDesc desc;
    desc.bufferType = STRUCTURED_BUFFER;
    desc.elementSize = sizeof(Vector3);
    desc.numElements = modelConstData.numVertices;
    desc.flags = DYNAMIC_BUFFER_FLAG;
    vertexPositionSB = Demo::renderer->CreateBuffer(desc, "Position");
    if(!vertexPositionSB)  
      return false;

    vertexNormalSB = Demo::renderer->CreateBuffer(desc, "Normal");
    if(!vertexNormalSB)  
      return false;

    vertexTangentSB = Demo::renderer->CreateBuffer(desc, "Tangent");
    if(!vertexTangentSB)  
      return false;
  }

  {
    // create pipeline state for skinning
    RootSignatureDesc rootSignatureDesc;
    // model info constant buffer
    rootSignatureDesc.rootParamDescs[0].rootParamType = CBV_ROOT_PARAM;
    rootSignatureDesc.rootParamDescs[0].rootDesc.shaderReg = 0;
    rootSignatureDesc.rootParamDescs[0].rootDesc.regSpace = 0;
    rootSignatureDesc.rootParamDescs[0].shaderVisibility = CS_SHADER_VIS;
    // weightIndexBuffer/ weightBuffer/ jointBuffer SRVs and positionBuffer/ normalBuffer/ tangentBuffer UAVs
    rootSignatureDesc.rootParamDescs[1].rootParamType = DESC_TABLE_ROOT_PARAM;
    rootSignatureDesc.rootParamDescs[1].rootDescTableDesc.numRanges = 2;
    rootSignatureDesc.rootParamDescs[1].rootDescTableDesc.ranges[0].rangeType = SRV_ROOT_DESC_TABLE_RANGE;
    rootSignatureDesc.rootParamDescs[1].rootDescTableDesc.ranges[0].numDescs = 3;
    rootSignatureDesc.rootParamDescs[1].rootDescTableDesc.ranges[0].baseShaderReg = 0;
    rootSignatureDesc.rootParamDescs[1].rootDescTableDesc.ranges[0].regSpace = 0;
    rootSignatureDesc.rootParamDescs[1].rootDescTableDesc.ranges[1].rangeType = UAV_ROOT_DESC_TABLE_RANGE;
    rootSignatureDesc.rootParamDescs[1].rootDescTableDesc.ranges[1].descTableOffset = 3;
    rootSignatureDesc.rootParamDescs[1].rootDescTableDesc.ranges[1].numDescs = 3;
    rootSignatureDesc.rootParamDescs[1].rootDescTableDesc.ranges[1].baseShaderReg = 0;
    rootSignatureDesc.rootParamDescs[1].rootDescTableDesc.ranges[1].regSpace = 0;
    rootSignatureDesc.rootParamDescs[1].shaderVisibility = CS_SHADER_VIS;
    rootSignatureDesc.numRootParamDescs = 2;
    DX12_RootSignature *rootSignature = Demo::renderer->CreateRootSignature(rootSignatureDesc, "Skinning");
    if(!rootSignature)
      return false;

    DX12_Shader *shader = Demo::resourceManager->LoadShader("shaders/skinning.sdr");
    if(!shader)
      return false;
 
    PipelineStateDesc pipelineStateDesc(COMPUTE_PIPELINE_STATE);
    pipelineStateDesc.rootSignature = rootSignature;
    pipelineStateDesc.compute.shader = shader;
    skinningPS = Demo::renderer->CreatePipelineState(pipelineStateDesc, "Skinning");
    if(!skinningPS)
      return false;
  }

  {
    // create constant buffer for storing model const data
    BufferDesc bufferDesc;
    bufferDesc.bufferType = CONSTANT_BUFFER;
    bufferDesc.elementSize = sizeof(ModelConstData);
    bufferDesc.numElements = 1;
    bufferDesc.flags = CPU_WRITE_BUFFER_FLAG | DYNAMIC_BUFFER_FLAG;
    modelCB = Demo::renderer->CreateBuffer(bufferDesc, "ModelConstData");
    if(!modelCB)
      return false;
  }

  // create descriptor table for skinning
  skinningDT.AddBufferSrv(weightIndexSB);
  skinningDT.AddBufferSrv(weightSB);
  skinningDT.AddBufferSrv(jointSB);
  skinningDT.AddBufferUav(vertexPositionSB);
  skinningDT.AddBufferUav(vertexNormalSB);
  skinningDT.AddBufferUav(vertexTangentSB);

  // create descriptor table for base pass vertex data
  basePassVertexDT.AddBufferSrv(vertexPositionSB);
  basePassVertexDT.AddBufferSrv(vertexNormalSB);
  basePassVertexDT.AddBufferSrv(vertexTangentSB);
  basePassVertexDT.AddBufferSrv(vertexUvHandednessSB);

  // create timer query for skinning pass
  skinningTQ = Demo::renderer->CreateTimerQuery("Skinning");
  if(!skinningTQ)
    return false;
  
  // create timer query for base pass
  basePassTQ = Demo::renderer->CreateTimerQuery("Base pass");
  if(!basePassTQ)
    return false;
  
  return true;
}

bool DemoModel::InitSkinnedDecals(const char** materialNames, UINT numMaterials)
{
  skinnedDecals = new SkinnedDecals;
  if(!skinnedDecals)
    return false;

  if(!skinnedDecals->Init(this, materialNames, numMaterials))
    return false;

  return true;
}

void DemoModel::UpdateSkeleton()
{
  Matrix4 transMatrix, xRotMatrix, yRotMatrix, zRotMatrix;
	transMatrix.SetTranslation(position);
	xRotMatrix.SetRotationY(rotation.x);
	yRotMatrix.SetRotationX(rotation.y);
	zRotMatrix.SetRotationZ(rotation.z);
	modelConstData.transformMatrix = transMatrix * zRotMatrix * yRotMatrix * xRotMatrix;

  nextAnimFrameIndex = pauseAnim ? currentAnimFrameIndex : (currentAnimFrameIndex + 1);	
  if(nextAnimFrameIndex > (numAnimFrames - 1))
    nextAnimFrameIndex = 0;
	if(animTimer.Update())
	  currentAnimFrameIndex = nextAnimFrameIndex;

  const AnimFrame &currentAnimFrame = animFrames[currentAnimFrameIndex];
  const AnimFrame &nextAnimFrame = animFrames[nextAnimFrameIndex];
  float timeFraction = static_cast<float>(animTimer.GetTimeFraction());

	for(UINT i=0; i<numJoints; i++) 
	{
		interpAnimFrame.joints[i].translation = currentAnimFrame.joints[i].translation.Lerp(nextAnimFrame.joints[i].translation, timeFraction);
    interpAnimFrame.joints[i].rotation = currentAnimFrame.joints[i].rotation.Slerp(nextAnimFrame.joints[i].rotation, timeFraction);
	}

  Aabb bounds = currentAnimFrame.bounds.Lerp(nextAnimFrame.bounds, timeFraction);
  bounds.GetTransformedAabb(interpAnimFrame.bounds, modelConstData.transformMatrix);

  const Camera *camera = Demo::renderer->GetCamera(MAIN_CAMERA_ID);
	visible = camera->GetFrustum().IsAabbInside(interpAnimFrame.bounds);
}

void DemoModel::UpdateBuffers()
{
  // update joint buffer
  jointSB->Update(interpAnimFrame.joints, numJoints);

  // update constant buffer
  modelConstData.debugDecalMask = debugDecalMask ? 1 : 0;
  modelCB->Update(&modelConstData, 1);
}

void DemoModel::PerformSkinning()
{
  DX12_CmdList *cmdList = Demo::renderer->GetCmdList(DEFAULT_CMD_LIST_ID);

  SCOPED_GPU_MARKER(cmdList, SKINNING_GPU_CMD_ORDER, "Skinning");
  SCOPED_TIMER_QUERY(cmdList, SKINNING_GPU_CMD_ORDER, skinningTQ);
 
  {
    ResourceBarrier barriers[3];
    barriers[0].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
    barriers[0].transition.resource = vertexPositionSB;
    barriers[0].transition.newResourceState = UNORDERED_ACCESS_RESOURCE_STATE;
    barriers[1].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
    barriers[1].transition.resource = vertexNormalSB;
    barriers[1].transition.newResourceState = UNORDERED_ACCESS_RESOURCE_STATE;
    barriers[2].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
    barriers[2].transition.resource = vertexTangentSB;
    barriers[2].transition.newResourceState = UNORDERED_ACCESS_RESOURCE_STATE;

    ResourceBarrierCmd cmd;
    cmd.barriers = barriers;
    cmd.numBarriers = _countof(barriers);
    cmdList->AddGpuCmd(cmd, SKINNING_GPU_CMD_ORDER);
  }

  {
    RootParam rootParams[2];
    rootParams[0].rootParamType = CBV_ROOT_PARAM;
    rootParams[0].bufferLocation = modelCB->GetGpuVirtualAddress();
    rootParams[1].rootParamType = DESC_TABLE_ROOT_PARAM;
    rootParams[1].baseGpuDescHandle = skinningDT.GetBaseDescHandle().gpuDescHandle;

    ComputeCmd cmd;
    cmd.numThreadGroupsX = (modelConstData.numVertices + SKINNING_THREAD_GROUP_SIZE - 1) / SKINNING_THREAD_GROUP_SIZE;
    cmd.numThreadGroupsY = 1;
    cmd.numThreadGroupsZ = 1;
    cmd.pipelineState = skinningPS;
    cmd.rootParams = rootParams;
    cmd.numRootParams = _countof(rootParams);
    cmdList->AddGpuCmd(cmd, SKINNING_GPU_CMD_ORDER);
  }

  {
    ResourceBarrier barriers[3];
    barriers[0].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
    barriers[0].transition.resource = vertexPositionSB;
    barriers[0].transition.newResourceState = NON_PIXEL_SHADER_RESOURCE_RESOURCE_STATE;
    barriers[1].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
    barriers[1].transition.resource = vertexNormalSB;
    barriers[1].transition.newResourceState = NON_PIXEL_SHADER_RESOURCE_RESOURCE_STATE;
    barriers[2].barrierType = TRANSITION_RESOURCE_BARRIER_TYPE;
    barriers[2].transition.resource = vertexTangentSB;
    barriers[2].transition.newResourceState = NON_PIXEL_SHADER_RESOURCE_RESOURCE_STATE;

    ResourceBarrierCmd cmd;
    cmd.barriers = barriers;
    cmd.numBarriers = _countof(barriers);
    cmdList->AddGpuCmd(cmd, SKINNING_GPU_CMD_ORDER);
  }
}

void DemoModel::RenderBasePass()
{
  DX12_CmdList *cmdList = Demo::renderer->GetCmdList(DEFAULT_CMD_LIST_ID);

  BEGIN_GPU_MARKER(cmdList, PRE_BASE_GPU_CMD_ORDER, "Base pass");
  BEGIN_TIMER_QUERY(cmdList, PRE_BASE_GPU_CMD_ORDER, basePassTQ);

  UINT decalMeshIndex = 0;
  for(UINT i=0; i<numSubModels; i++)
  {
    const bool decalsEnabled = subModels[i].material->receiveDecals && useDecals;

    CpuDescHandle rtvCpuDescHandles[] = 
    {
      Demo::renderer->GetRenderTarget(ACCUM_BUFFER_RT_ID)->GetRtv().cpuDescHandle
    };

    RootParam rootParams[9];
    rootParams[0].rootParamType = CBV_ROOT_PARAM;
    rootParams[0].bufferLocation = Demo::renderer->GetCamera(MAIN_CAMERA_ID)->GetConstantBuffer()->GetGpuVirtualAddress();
    rootParams[1].rootParamType = CBV_ROOT_PARAM;
    rootParams[1].bufferLocation = shadingPP->GetLightingCB()->GetGpuVirtualAddress();
    rootParams[2].rootParamType = CBV_ROOT_PARAM;
    rootParams[2].bufferLocation = modelCB->GetGpuVirtualAddress();
    rootParams[3].rootParamType = DESC_TABLE_ROOT_PARAM;
    rootParams[3].baseGpuDescHandle = basePassVertexDT.GetBaseDescHandle().gpuDescHandle;
    rootParams[4].rootParamType = DESC_TABLE_ROOT_PARAM;
    rootParams[4].baseGpuDescHandle = subModels[i].materialDT.GetBaseDescHandle().gpuDescHandle;
    if(decalsEnabled)
    {
      assert(skinnedDecals != nullptr);
      rootParams[5].rootParamType = CONST_ROOT_PARAM;
      rootParams[5].rootConst.constData[0] = decalMeshIndex++;
      rootParams[5].rootConst.numConsts = 1;
      rootParams[6].rootParamType = CBV_ROOT_PARAM;
      rootParams[6].bufferLocation = skinnedDecals->decalCB->GetGpuVirtualAddress();
      rootParams[7].rootParamType = DESC_TABLE_ROOT_PARAM;
      rootParams[7].baseGpuDescHandle = skinnedDecals->decalBuffersDT.GetBaseDescHandle().gpuDescHandle;
      rootParams[8].rootParamType = DESC_TABLE_ROOT_PARAM;
      rootParams[8].baseGpuDescHandle = skinnedDecals->decalMaterialsDT.GetBaseDescHandle().gpuDescHandle;
    }
  
    DrawCmd cmd;
    cmd.rtvCpuDescHandles = rtvCpuDescHandles;
    cmd.numRenderTargets = _countof(rtvCpuDescHandles);
    cmd.dsvCpuDescHandle = Demo::renderer->GetDepthStencilTarget(MAIN_DEPTH_DST_ID)->GetDsv(DEFAULT_DSV_TYPE).cpuDescHandle;
    cmd.viewportSet = Demo::renderer->GetViewportSet(DEFAULT_VIEWPORT_SET_ID);
    cmd.scissorRectSet = Demo::renderer->GetScissorRectSet(DEFAULT_SCISSOR_RECT_SET_ID);
    cmd.indexBuffer = indexBuffer;
    cmd.primitiveTopology = TRIANGLELIST_PRIMITIVE_TOPOLOGY;
    cmd.firstIndex = subModels[i].firstIndex;
    cmd.numElements = subModels[i].numIndices;
    cmd.pipelineState = decalsEnabled ? subModels[i].baseWithDecalsPS : subModels[i].baseNoDecalsPS;
    cmd.rootParams = rootParams;
    cmd.numRootParams = decalsEnabled ? 9 : 5;
    cmdList->AddGpuCmd(cmd, subModels[i].material->alphaTested ? BASE_ALPHA_TESTED_GPU_CMD_ORDER : BASE_GPU_CMD_ORDER);
  }

  END_TIMER_QUERY(cmdList, POST_BASE_GPU_CMD_ORDER, basePassTQ);
  END_GPU_MARKER(cmdList, POST_BASE_GPU_CMD_ORDER);
}

void DemoModel::Render()
{
	UpdateSkeleton();

  bool skinningDataRequired = visible;
  if(skinnedDecals)
  {
    skinningDataRequired = skinningDataRequired || skinnedDecals->RequiresSkinningData();
  }

  if(skinningDataRequired)
  {
    UpdateBuffers();
    PerformSkinning();
  }

  if(skinnedDecals)
    skinnedDecals->Render();

  if(visible)
	  RenderBasePass();
}
