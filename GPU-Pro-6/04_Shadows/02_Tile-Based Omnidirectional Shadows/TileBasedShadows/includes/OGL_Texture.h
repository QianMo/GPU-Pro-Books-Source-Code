#ifndef OGL_TEXTURE_H
#define OGL_TEXTURE_H

#include <render_states.h>
#include <Image.h>
#include <OGL_Sampler.h>

struct OGL_TexFormat
{
  GLint internalFormat; // internal OpenGL texture-format
  GLenum srcFormat; // format of source-image
  GLenum srcType; // data-type of source-image
};

// OGL_Texture
//
// Manages a texture.
class OGL_Texture 
{
public:
  friend class OGL_RenderTarget; 

  OGL_Texture():
    textureName(0),
    target(GL_TEXTURE_2D),
    format(TEX_FORMAT_NONE)
  {
    name[0] = 0;
  }

  ~OGL_Texture()
  {
    Release();
  }

  void Release();	

  bool LoadFromFile(const char *fileName); 

  // creates render-target texture
  bool CreateRenderable(unsigned int width, unsigned int height, unsigned int depth, texFormats format, unsigned int rtFlags=0);	

  void Bind(textureBP bindingPoint) const;

  const char* GetName() const
  {
    return name;
  }

  static OGL_TexFormat GetOglTexFormat(texFormats texFormat);

private:
  char name[DEMO_MAX_FILENAME];
  GLuint textureName; 
  GLenum target;
  texFormats format;

};

#endif
