#include <stdafx.h>
#include <Demo.h>
#include <GuiManager.h>

#define IMGUI_MAX_VERTEX_COUNT 2048
#define IMGUI_MAX_INDEX_COUNT 4096

void GuiManager::Release()
{
  ImGui::Shutdown();
}

bool GuiManager::Init()
{
  ImGuiIO &io = ImGui::GetIO();
  io.IniFilename = nullptr;
  io.KeyMap[ImGuiKey_Tab] = VK_TAB;
  io.KeyMap[ImGuiKey_LeftArrow] = VK_LEFT;
  io.KeyMap[ImGuiKey_RightArrow] = VK_RIGHT;
  io.KeyMap[ImGuiKey_UpArrow] = VK_UP;
  io.KeyMap[ImGuiKey_DownArrow] = VK_DOWN;
  io.KeyMap[ImGuiKey_PageUp] = VK_PRIOR;
  io.KeyMap[ImGuiKey_PageDown] = VK_NEXT;
  io.KeyMap[ImGuiKey_Home] = VK_HOME;
  io.KeyMap[ImGuiKey_End] = VK_END;
  io.KeyMap[ImGuiKey_Delete] = VK_DELETE;
  io.KeyMap[ImGuiKey_Backspace] = VK_BACK;
  io.KeyMap[ImGuiKey_Enter] = VK_RETURN;
  io.KeyMap[ImGuiKey_Escape] = VK_ESCAPE;
  io.RenderDrawListsFn = nullptr;
  io.ImeWindowHandle = Demo::window->GetHWnd();

  RootSignatureDesc rootSignatureDesc; 
  rootSignatureDesc.numRootParamDescs = 2;
  rootSignatureDesc.rootParamDescs[0].rootParamType = CBV_ROOT_PARAM;
  rootSignatureDesc.rootParamDescs[0].rootDesc.shaderReg = 0;
  rootSignatureDesc.rootParamDescs[0].rootDesc.regSpace = 0;
  rootSignatureDesc.rootParamDescs[0].shaderVisibility = VS_SHADER_VIS;
  rootSignatureDesc.rootParamDescs[1].rootParamType = DESC_TABLE_ROOT_PARAM;
  rootSignatureDesc.rootParamDescs[1].rootDescTableDesc.numRanges = 1;
  rootSignatureDesc.rootParamDescs[1].rootDescTableDesc.ranges[0].rangeType = SRV_ROOT_DESC_TABLE_RANGE;
  rootSignatureDesc.rootParamDescs[1].rootDescTableDesc.ranges[0].numDescs = 1;
  rootSignatureDesc.rootParamDescs[1].rootDescTableDesc.ranges[0].baseShaderReg = 0;
  rootSignatureDesc.rootParamDescs[1].rootDescTableDesc.ranges[0].regSpace = 0;
  rootSignatureDesc.rootParamDescs[1].shaderVisibility = PS_SHADER_VIS;
  rootSignatureDesc.numSamplerDescs = 1;
  rootSignatureDesc.samplerDescs[0].filter = MIN_MAG_LINEAR_MIP_POINT_FILTER;
  rootSignatureDesc.samplerDescs[0].shaderReg = 0;
  rootSignatureDesc.samplerDescs[0].regSpace = 0;
  rootSignatureDesc.samplerDescs[0].shaderVisibility = PS_SHADER_VIS;
  rootSignatureDesc.flags = ALLOW_INPUT_LAYOUT_ROOT_SIGNATURE_FLAG;
  DX12_RootSignature *rootSignature = Demo::renderer->CreateRootSignature(rootSignatureDesc, "GUI");
  if(!rootSignature)
    return false;

  DX12_Shader *shader = Demo::resourceManager->LoadShader("shaders/imgui.sdr");
  if(!shader)
    return false;

  PipelineStateDesc pipelineStateDesc(GRAPHICS_PIPELINE_STATE);
  pipelineStateDesc.rootSignature = rootSignature;
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
  pipelineStateDesc.graphics.shader = shader;
  pipelineStateDesc.graphics.numRenderTargets = 1;
  pipelineStateDesc.graphics.rtvFormats[0] = Demo::renderer->GetRenderTarget(BACK_BUFFER_RT_ID)->GetTexture()->GetTextureDesc().format;
  pipelineStateDesc.graphics.depthStencilDesc.depthTest = false;
  pipelineStateDesc.graphics.depthStencilDesc.depthMask = false;
  pipelineStateDesc.graphics.blendDesc.blendEnable = true;
  pipelineStateDesc.graphics.blendDesc.srcColorBlend = SRC_ALPHA_BLEND;
  pipelineStateDesc.graphics.blendDesc.dstColorBlend = INV_SRC_ALPHA_BLEND;
  pipelineState = Demo::renderer->CreatePipelineState(pipelineStateDesc, "GUI");
  if(!pipelineState)
    return false;

  unsigned char* pixels = nullptr;
  int texWidth = 0;
  int texHeight = 0;
  io.Fonts->GetTexDataAsRGBA32(&pixels, &texWidth, &texHeight);
  ImageDesc imageDesc;
  imageDesc.name = "imgui";
  imageDesc.width = texWidth;
  imageDesc.height = texHeight;
  imageDesc.depth = 1;
  imageDesc.numMipMaps = 1;
  imageDesc.format = RGBA8_RENDER_FORMAT;
  imageDesc.data = pixels;
  Image image;
  image.Create(imageDesc);
  const Image *imagePtr = &image;
  DX12_Texture *texture = Demo::resourceManager->CreateTexture(&imagePtr, 1, STATIC_TEXTURE_FLAG);
  if(!texture)
    return false;
  io.Fonts->TexID = &texture;
  descTable.AddTextureSrv(texture);

  {
    BufferDesc bufferDesc;
    bufferDesc.bufferType = VERTEX_BUFFER;
    bufferDesc.elementSize = sizeof(ImDrawVert);
    bufferDesc.numElements = IMGUI_MAX_VERTEX_COUNT;
    bufferDesc.flags = CPU_WRITE_BUFFER_FLAG | DYNAMIC_BUFFER_FLAG;
    vertexBuffer = Demo::renderer->CreateBuffer(bufferDesc, "imgui");
    if(!vertexBuffer)
      return false;
  }

  {
    BufferDesc bufferDesc;
    bufferDesc.bufferType = INDEX_BUFFER;
    bufferDesc.elementSize = sizeof(unsigned short);
    bufferDesc.numElements = IMGUI_MAX_INDEX_COUNT;
    bufferDesc.elementFormat = R16UI_RENDER_FORMAT;
    bufferDesc.flags = CPU_WRITE_BUFFER_FLAG | DYNAMIC_BUFFER_FLAG;
    indexBuffer = Demo::renderer->CreateBuffer(bufferDesc, "imgui");
    if(!indexBuffer)
      return false;
  }

  {
    BufferDesc bufferDesc;
    bufferDesc.bufferType = CONSTANT_BUFFER;
    bufferDesc.elementSize = sizeof(Matrix4);
    bufferDesc.numElements = 1;
    bufferDesc.flags = CPU_WRITE_BUFFER_FLAG | DYNAMIC_BUFFER_FLAG;
    constantBuffer = Demo::renderer->CreateBuffer(bufferDesc, "imgui");
    if(!constantBuffer)
      return false;
  }

  if(!vertices.Resize(IMGUI_MAX_VERTEX_COUNT))
    return false;

  if(!indices.Resize(IMGUI_MAX_INDEX_COUNT))
    return false;

  return true;
}

void GuiManager::MessageCallback(UINT uMsg, WPARAM wParam, LPARAM lParam)
{
  ImGuiIO &io = ImGui::GetIO();
  switch(uMsg)
  {
  case WM_LBUTTONDOWN:
    io.MouseDown[0] = true;
    return;

  case WM_LBUTTONUP:
    io.MouseDown[0] = false;
    return;

  case WM_RBUTTONDOWN:
    io.MouseDown[1] = true;
    return;

  case WM_RBUTTONUP:
    io.MouseDown[1] = false;
    return;

  case WM_MOUSEMOVE:
    io.MousePos.x = static_cast<signed short>(lParam);
    io.MousePos.y = static_cast<signed short>(lParam >> 16);
    return;

  case WM_KEYDOWN:
    if(wParam < 256)
      io.KeysDown[wParam] = 1;
    return;

  case WM_KEYUP:
    if(wParam < 256)
      io.KeysDown[wParam] = 0;
    return;

  case WM_CHAR:
    if(wParam > 0 && wParam < 0x10000)
      io.AddInputCharacter(static_cast<unsigned short>(wParam));
    return;
  }
}

void GuiManager::BeginFrame()
{
  ImGuiIO &io = ImGui::GetIO();

  io.DisplaySize = ImVec2(float(SCREEN_WIDTH), float(SCREEN_HEIGHT));
  io.DeltaTime = static_cast<float>(Demo::timeManager->GetFrameInterval()) * 0.001f;
 
  io.KeyCtrl = (GetKeyState(VK_CONTROL) & 0x8000) != 0;
  io.KeyShift = (GetKeyState(VK_SHIFT) & 0x8000) != 0;
  io.KeyAlt = (GetKeyState(VK_MENU) & 0x8000) != 0;

  ImGui::NewFrame();
  vertices.Clear();
  indices.Clear();
}

void GuiManager::EndFrame()
{
  ImGui::Render();

  const ImDrawData *drawData = ImGui::GetDrawData();

  assert(drawData->TotalVtxCount <= IMGUI_MAX_VERTEX_COUNT);
  assert(drawData->TotalIdxCount <= IMGUI_MAX_INDEX_COUNT);
  for(int i=0; i<drawData->CmdListsCount; i++)
  {
    const ImDrawList *drawList = drawData->CmdLists[i];
    vertices.AddElements(drawList->VtxBuffer.size(), &drawList->VtxBuffer[0]);
    indices.AddElements(drawList->IdxBuffer.size(), &drawList->IdxBuffer[0]);
  }
  vertexBuffer->Update(vertices, vertices.GetSize());
  indexBuffer->Update(indices, indices.GetSize());

  Matrix4 projMatrix;
  projMatrix.entries[0] = 2.0f / SCREEN_WIDTH;
  projMatrix.entries[5] = -2.0f / SCREEN_HEIGHT;
  projMatrix.entries[10] = 0.5f;
  projMatrix.entries[12] = -1.0f;
  projMatrix.entries[13] = 1.0f;
  projMatrix.entries[14] = 0.5f;
  constantBuffer->Update(&projMatrix, 1);

  DX12_CmdList *cmdList = Demo::renderer->GetCmdList(FINAL_CMD_LIST_ID);

  SCOPED_GPU_MARKER(cmdList, GUI_GPU_CMD_ORDER, "GUI");

  UINT vertexOffset = 0;
  UINT indexOffset = 0;
  for(int i=0; i<drawData->CmdListsCount; i++)
  {
    const ImDrawList *drawList = drawData->CmdLists[i];
    for(int j=0; j<drawList->CmdBuffer.size(); j++)
    {
      const ImDrawCmd* drawCmd = &drawList->CmdBuffer[j];
      assert(drawCmd->UserCallback == false);

      CpuDescHandle rtvCpuDescHandles[1] = 
      {
        Demo::renderer->GetRenderTarget(BACK_BUFFER_RT_ID)->GetRtv().cpuDescHandle
      };

      DX12_Buffer *vertexBuffers[1];
      vertexBuffers[0] = vertexBuffer;

      RootParam rootParams[2];
      rootParams[0].rootParamType = CBV_ROOT_PARAM;
      rootParams[0].bufferLocation = constantBuffer->GetGpuVirtualAddress();
      rootParams[1].rootParamType = DESC_TABLE_ROOT_PARAM;
      rootParams[1].baseGpuDescHandle = descTable.GetBaseDescHandle().gpuDescHandle;

      DrawCmd cmd;
      cmd.rtvCpuDescHandles = rtvCpuDescHandles;
      cmd.numRenderTargets = _countof(rtvCpuDescHandles);
      cmd.viewportSet = Demo::renderer->GetViewportSet(DEFAULT_VIEWPORT_SET_ID);
      cmd.scissorRectSet = Demo::renderer->GetScissorRectSet(DEFAULT_SCISSOR_RECT_SET_ID);
      cmd.vertexBuffers = vertexBuffers;
      cmd.numVertexBuffers = _countof(vertexBuffers);
      cmd.indexBuffer = indexBuffer;
      cmd.primitiveTopology = TRIANGLELIST_PRIMITIVE_TOPOLOGY;
      cmd.firstIndex = indexOffset;
      cmd.baseVertexIndex = vertexOffset;
      cmd.numElements = drawCmd->ElemCount;
      cmd.pipelineState = pipelineState;
      cmd.rootParams = rootParams;
      cmd.numRootParams = _countof(rootParams);
      cmdList->AddGpuCmd(cmd, GUI_GPU_CMD_ORDER);
      indexOffset += drawCmd->ElemCount;
    }
    vertexOffset += drawList->VtxBuffer.size();
  }
}

bool GuiManager::BeginWindow(const char *name, const Vector2 &position, const Vector2 &size)
{
  ImGui::SetNextWindowPos(ImVec2(position.x, position.y), ImGuiSetCond_Once);
  ImGui::SetNextWindowSize(ImVec2(size.x, size.y), ImGuiSetCond_Once);
  return ImGui::Begin(name);
}

void GuiManager::EndWindow()
{
  ImGui::End();
}

bool GuiManager::ShowComboBox(const char *name, float width, const char **items, UINT numItems, UINT &currentItem)
{
  ImGui::PushItemWidth(width);
  bool valueChanged = ImGui::Combo(name, reinterpret_cast<int*>(&currentItem), items, numItems);
  CLAMP(currentItem, 0, numItems-1);
  return valueChanged;
}

bool GuiManager::ShowSliderInt(const char *name, float width, int minValue, int maxValue, int &currentValue)
{
  ImGui::PushItemWidth(width);
  bool valueChanged = ImGui::SliderInt(name, &currentValue, minValue, maxValue);
  CLAMP(currentValue, minValue, maxValue);
  return valueChanged;
}

bool GuiManager::ShowSliderFloat(const char *name, float width, float minValue, float maxValue, float &currentValue)
{
  ImGui::PushItemWidth(width);
  bool valueChanged = ImGui::SliderFloat(name, &currentValue, minValue, maxValue);
  CLAMP(currentValue, minValue, maxValue);
  return valueChanged;
}

bool GuiManager::ShowCheckBox(const char *name, bool &currentValue)
{
  return ImGui::Checkbox(name, &currentValue);
}

void GuiManager::ShowText(const char *text, ...)
{
  va_list args;
  va_start(args, text);
  ImGui::TextV(text, args);
  va_end(args);
}