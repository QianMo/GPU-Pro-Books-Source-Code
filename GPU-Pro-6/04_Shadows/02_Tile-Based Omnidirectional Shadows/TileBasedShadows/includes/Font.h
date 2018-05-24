#ifndef FONT_H
#define FONT_H

#define CURRENT_FONT_VERSION 1
#define FONT_MAX_TEXT_LENGTH 1024 // max length of text, which can be each time printed
#define FONT_MAX_VERTEX_COUNT 4096 // max number of vertices, that font can render

class RenderTargetConfig;
class OGL_VertexLayout;
class OGL_VertexBuffer;
class Material;

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
    texCoords(NULL),
    material(NULL),
    vertexLayout(NULL),
    vertexBuffer(NULL)
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

  void AddSurfaces();  

  const char* GetName() const
  { 
    return name;
  }

private:
  char name[DEMO_MAX_FILENAME];

  unsigned int textureWidth, textureHeight;	// width/ height of font-texture
  unsigned int fontHeight; // height of used font
  unsigned int fontSpacing; // spacing of used font
  unsigned int numTexCoords; // number of texCoords
  float *texCoords; // texCoords for fetching corresponding part for each character from font-texture
  
  Material *material;
  OGL_VertexLayout *vertexLayout;
  OGL_VertexBuffer *vertexBuffer;

};

#endif
