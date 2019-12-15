#include <stdafx.h>
#include <Demo.h>
#include <Font.h>

#define CURRENT_FONT_VERSION 1
#define FONT_MAX_TEXT_LENGTH 1024 // max length of text, which can be each time printed
#define FONT_MAX_VERTEX_COUNT 8192 // max number of vertices, that font can render

void Font::Release()
{
  SAFE_DELETE_ARRAY(texCoords);
}

bool Font::Load(const char *fileName)
{
  // load ".font" file
  strcpy(name, fileName);
  char filePath[DEMO_MAX_FILEPATH];
  Demo::fileManager->GetFilePath(fileName, filePath);
  FILE *file;
  fopen_s(&file, filePath, "rb");
  if(!file)
    return false;

  // check idString
  char idString[10];
  memset(idString, 0, 10);
  fread(idString, sizeof(char), 9, file);
  if(strcmp(idString, "DEMO_FONT") != 0)
  {
    fclose(file);
    return false;
  }

  // check version
  UINT version;
  fread(&version, sizeof(UINT), 1, file);
  if(version != CURRENT_FONT_VERSION)
  {
    fclose(file);
    return false;
  }

  // load material
  char fontMaterialName[256];
  fread(fontMaterialName, sizeof(char), 256, file);
  material = Demo::resourceManager->LoadMaterial(fontMaterialName);
  if(!material)
  {
    fclose(file);
    return false;
  }
  descTable.AddTextureSrv(material->textures[COLOR_TEX_ID]);

  // load font parameters 
  fread(&textureWidth, sizeof(UINT), 1, file);
  fread(&textureHeight, sizeof(UINT), 1, file);
  fread(&fontHeight, sizeof(UINT), 1, file);
  fread(&fontSpacing, sizeof(UINT), 1, file);

  // get number of texCoords
  fread(&numTexCoords, sizeof(UINT), 1, file);
  if(numTexCoords < 1)
  {
    fclose(file);
    return false;
  }

  // load texCoords
  texCoords = new float[numTexCoords];
  if(!texCoords)
  {
    fclose(file);
    return false;
  }
  fread(texCoords, sizeof(float), numTexCoords, file);

  fclose(file);

  backBufferRT = Demo::renderer->GetRenderTarget(BACK_BUFFER_RT_ID);
  if(!backBufferRT)
  {
    SAFE_DELETE_ARRAY(texCoords);
    return false;
  }

  RootSignatureDesc rootSignatureDesc;
  rootSignatureDesc.numRootParamDescs = 1;
  rootSignatureDesc.rootParamDescs[0].rootParamType = DESC_TABLE_ROOT_PARAM;
  rootSignatureDesc.rootParamDescs[0].rootDescTableDesc.numRanges = 1;
  rootSignatureDesc.rootParamDescs[0].rootDescTableDesc.ranges[0].rangeType = SRV_ROOT_DESC_TABLE_RANGE;
  rootSignatureDesc.rootParamDescs[0].rootDescTableDesc.ranges[0].numDescs = 1;
  rootSignatureDesc.rootParamDescs[0].rootDescTableDesc.ranges[0].baseShaderReg = 0;
  rootSignatureDesc.rootParamDescs[0].rootDescTableDesc.ranges[0].regSpace = 0;
  rootSignatureDesc.rootParamDescs[0].shaderVisibility = PS_SHADER_VIS;
  rootSignatureDesc.numSamplerDescs = 1;
  rootSignatureDesc.samplerDescs[0].filter = MIN_MAG_MIP_LINEAR_FILTER;
  rootSignatureDesc.samplerDescs[0].shaderReg = 0;
  rootSignatureDesc.samplerDescs[0].regSpace = 0;
  rootSignatureDesc.samplerDescs[0].shaderVisibility = PS_SHADER_VIS;
  rootSignatureDesc.flags = ALLOW_INPUT_LAYOUT_ROOT_SIGNATURE_FLAG;
  DX12_RootSignature *rootSignature = Demo::renderer->CreateRootSignature(rootSignatureDesc, "Font");
  if(!rootSignature)
  {
    SAFE_DELETE_ARRAY(texCoords);
    return false;
  }

  PipelineStateDesc pipelineStateDesc(GRAPHICS_PIPELINE_STATE);
  pipelineStateDesc.rootSignature = rootSignature;
  pipelineStateDesc.graphics.primitiveTopologyType = TRIANGLE_PRIMITIVE_TOPOLOGY_TYPE;
  pipelineStateDesc.graphics.inputLayout.numElementDescs = 3;
  pipelineStateDesc.graphics.inputLayout.elementDescs[0].inputElement = POSITION_INPUT_ELEMENT;
  pipelineStateDesc.graphics.inputLayout.elementDescs[0].format = RG32F_RENDER_FORMAT;
  pipelineStateDesc.graphics.inputLayout.elementDescs[0].offset = 0;
  pipelineStateDesc.graphics.inputLayout.elementDescs[1].inputElement = TEXCOORDS_INPUT_ELEMENT;
  pipelineStateDesc.graphics.inputLayout.elementDescs[1].format = RG32F_RENDER_FORMAT;
  pipelineStateDesc.graphics.inputLayout.elementDescs[1].offset = 8;
  pipelineStateDesc.graphics.inputLayout.elementDescs[2].inputElement = COLOR_INPUT_ELEMENT;
  pipelineStateDesc.graphics.inputLayout.elementDescs[2].format = RGBA8_RENDER_FORMAT;
  pipelineStateDesc.graphics.inputLayout.elementDescs[2].offset = 16;
  pipelineStateDesc.graphics.shader = material->shader;
  pipelineStateDesc.graphics.numRenderTargets = 1;
  pipelineStateDesc.graphics.rtvFormats[0] = RenderFormat::ConvertToSrgbFormat(backBufferRT->GetTexture()->GetTextureDesc().format);
  pipelineStateDesc.graphics.blendDesc = material->blendDesc;
  pipelineStateDesc.graphics.rasterizerDesc = material->rasterDesc;
  pipelineStateDesc.graphics.depthStencilDesc = material->depthStencilDesc;
  pipelineState = Demo::renderer->CreatePipelineState(pipelineStateDesc, "Font");
  if(!pipelineState)
  {
    SAFE_DELETE_ARRAY(texCoords);
    return false;
  }

  BufferDesc bufferDesc;
  bufferDesc.bufferType = VERTEX_BUFFER;
  bufferDesc.elementSize = sizeof(FontVertex);
  bufferDesc.numElements = FONT_MAX_VERTEX_COUNT;
  bufferDesc.flags = CPU_WRITE_BUFFER_FLAG | DYNAMIC_BUFFER_FLAG;
  vertexBuffer = Demo::renderer->CreateBuffer(bufferDesc, "Font");
  if(!vertexBuffer)
  {
    SAFE_DELETE_ARRAY(texCoords);
    return false;
  }

  if(!vertices.Resize(FONT_MAX_VERTEX_COUNT))
    return false;

  return true;
}

void Font::Print(const Vector2 &position, float scale, const Color &color, const char *string, ...)
{
  char str[FONT_MAX_TEXT_LENGTH];
  va_list va;
  if(!string)
    return;
  va_start(va, string);
  UINT length = _vscprintf(string, va) + 1;
  if(length > FONT_MAX_TEXT_LENGTH)
  {
    va_end(va);
    return;
  }
  vsprintf_s(str, string, va);
  va_end(va);

  char *text = str;
  float positionX = position.x;
  float positionY = position.y;
  positionX -= (float)(fontSpacing / fontHeight) + (scale*0.5f);
  const float startX = positionX;
  float aspectRatio = Demo::renderer->GetCamera(MAIN_CAMERA_ID)->GetAspectRatio();
  const float scaleX = scale;
  const float scaleY = scale*aspectRatio;
  const int maxCharIndex = numTexCoords / 4;
  UINT fontColor = color.GetAsRgba8();
  FontVertex newVertices[6];
  while(*text)
  {
    char c = *text++;
    if(c == '\n')
    {
      positionX = startX;
      positionY -= ((texCoords[3] - texCoords[1])*textureHeight / (float)fontHeight)*scaleY;
    }
    int charIndex = c - 32;
    if((charIndex < 0) || (charIndex >= maxCharIndex))
      continue;
    float tx1 = texCoords[charIndex * 4];
    float ty1 = texCoords[charIndex * 4 + 3];
    float tx2 = texCoords[charIndex * 4 + 2];
    float ty2 = texCoords[charIndex * 4 + 1];
    float width = ((tx2 - tx1)*textureWidth / (float)fontHeight)*scaleX;
    float height = ((ty1 - ty2)*textureHeight / (float)fontHeight)*scaleY;
    if(c != ' ')
    {
      newVertices[0].position = Vector2(positionX, positionY);
      newVertices[0].texCoords = Vector2(tx1, ty1);
      newVertices[0].color = fontColor;
      newVertices[1].position = Vector2(positionX + width, positionY);
      newVertices[1].texCoords = Vector2(tx2, ty1);
      newVertices[1].color = fontColor;
      newVertices[2].position = Vector2(positionX, positionY + height);
      newVertices[2].texCoords = Vector2(tx1, ty2);
      newVertices[2].color = fontColor;
      newVertices[3].position = Vector2(positionX + width, positionY);
      newVertices[3].texCoords = Vector2(tx2, ty1);
      newVertices[3].color = fontColor;
      newVertices[4].position = Vector2(positionX + width, positionY + height);
      newVertices[4].texCoords = Vector2(tx2, ty2);
      newVertices[4].color = fontColor;
      newVertices[5].position = Vector2(positionX, positionY + height);
      newVertices[5].texCoords = Vector2(tx1, ty2);
      newVertices[5].color = fontColor;
      vertices.AddElements(6, newVertices);
    }

    positionX += width - (2.0f*fontSpacing*scaleX) / (float)fontHeight;
  }
}

void Font::Render()
{
  vertexBuffer->Update(vertices, vertices.GetSize());
 
  DX12_CmdList *cmdList = Demo::renderer->GetCmdList(FINAL_CMD_LIST_ID);

  SCOPED_GPU_MARKER(cmdList, GUI_GPU_CMD_ORDER, "Font");

  {
    CpuDescHandle rtvCpuDescHandles[] = 
    {
      backBufferRT->GetRtv(SRGB_RTV_TYPE).cpuDescHandle
    };

    DX12_Buffer *vertexBuffers[1];
    vertexBuffers[0] = vertexBuffer;

    RootParam rootParams[1];
    rootParams[0].rootParamType = DESC_TABLE_ROOT_PARAM;
    rootParams[0].baseGpuDescHandle = descTable.GetBaseDescHandle().gpuDescHandle;

    DrawCmd cmd;
    cmd.rtvCpuDescHandles = rtvCpuDescHandles;
    cmd.numRenderTargets = _countof(rtvCpuDescHandles);
    cmd.viewportSet = Demo::renderer->GetViewportSet(DEFAULT_VIEWPORT_SET_ID);
    cmd.scissorRectSet = Demo::renderer->GetScissorRectSet(DEFAULT_SCISSOR_RECT_SET_ID);
    cmd.vertexBuffers = vertexBuffers;
    cmd.numVertexBuffers = _countof(vertexBuffers);
    cmd.primitiveTopology = TRIANGLELIST_PRIMITIVE_TOPOLOGY;
    cmd.firstIndex = 0;
    cmd.numElements = vertices.GetSize();
    cmd.pipelineState = pipelineState;
    cmd.rootParams = rootParams;
    cmd.numRootParams = _countof(rootParams);
    cmdList->AddGpuCmd(cmd, GUI_GPU_CMD_ORDER);
  }

  vertices.Clear();
}

