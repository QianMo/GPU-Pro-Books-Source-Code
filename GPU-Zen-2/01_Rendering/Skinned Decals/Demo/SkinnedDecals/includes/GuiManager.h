#ifndef GUI_MANAGER_H
#define GUI_MANAGER_H

#include <List.h>
#include <DX12_ResourceDescTable.h>

class DX12_PipelineState;
class DX12_Buffer;

// GuiManager
//
// Manager for imgui.
class GuiManager
{
public:
  GuiManager():
    pipelineState(nullptr),
    vertexBuffer(nullptr),
    indexBuffer(nullptr),
    constantBuffer(nullptr)
  {
  }

  ~GuiManager()
  {
    Release();
  }

  void Release();

  bool Init();

  void MessageCallback(UINT uMsg, WPARAM wParam, LPARAM lParam);

  void BeginFrame();

  void EndFrame();

  bool BeginWindow(const char *name, const Vector2 &position, const Vector2 &size);

  void EndWindow();

  bool ShowComboBox(const char *name, float width, const char **items, UINT numItems, UINT &currentItem);

  bool ShowSliderInt(const char *name, float width, int minValue, int maxValue, int &currentValue);

  bool ShowSliderFloat(const char *name, float width, float minValue, float maxValue, float &currentValue);

  bool ShowCheckBox(const char *name, bool &currentValue);

  void ShowText(const char *text, ...);

private:
  DX12_PipelineState *pipelineState;
  DX12_Buffer *vertexBuffer;
  DX12_Buffer *indexBuffer;
  DX12_Buffer *constantBuffer;
  DX12_ResourceDescTable descTable;

  List<ImDrawVert>vertices;
  List<unsigned short>indices;

};

#endif