#ifndef FONT_H
#define FONT_H

#include <List.h>
#include <DX12_ResourceDescTable.h>

class Material;
class DX12_RenderTarget;
class DX12_PipelineState;
class DX12_Buffer;

struct FontVertex
{
  Vector2 position;
  Vector2 texCoords;
  UINT color;
}; 

// Font
//
// Texture-based font, that uses a pre-generated texture, in which a set of all required characters 
// is stored. All information (used material, font parameters, texCoords) are stored in a simple 
// custom file-format (".font").
// When outputting text, every character is looked up in the font-texture by texture coordinates and 
// the corresponding part of the font-texture is mapped onto a quad. All text from the same font is 
// collected and rendered in a single draw-batch.
class Font
{
public:
  Font():
    textureWidth(0),
    textureHeight(0),
    fontHeight(0),
    fontSpacing(0),
    numTexCoords(0),
    texCoords(nullptr),
    material(nullptr),
    backBufferRT(nullptr),
    pipelineState(nullptr),
    vertexBuffer(nullptr)
  {
    name[0] = 0;
  }

  ~Font()
  {
    Release();
  }

  void Release();

  bool Load(const char *fileName);

  void Print(const Vector2 &position, float scale, const Color &color, const char *string, ...);

  void Render();

  const char* GetName() const
  {
    return name;
  }

private:
  char name[DEMO_MAX_FILENAME];

  UINT textureWidth, textureHeight;	// width/ height of font-texture
  UINT fontHeight; // height of used font
  UINT fontSpacing; // spacing of used font
  UINT numTexCoords; // number of texCoords
  float *texCoords; // texCoords for fetching corresponding part for each character from font-texture
  List<FontVertex> vertices;

  Material *material;
  DX12_RenderTarget *backBufferRT;
  DX12_PipelineState *pipelineState;
  DX12_Buffer *vertexBuffer;
  DX12_ResourceDescTable descTable;

};

#endif
