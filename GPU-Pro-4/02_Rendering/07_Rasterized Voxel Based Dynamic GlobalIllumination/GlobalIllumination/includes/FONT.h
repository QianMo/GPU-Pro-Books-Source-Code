#ifndef FONT_H
#define FONT_H

#define CURRENT_FONT_VERSION 1
#define FONT_MAX_TEXT_LENGTH 1024 // max length of text, which can be outputted at once
#define FONT_MAX_VERTEX_COUNT 4096 // max number of vertices, that font can render

class RENDER_TARGET_CONFIG;
class DX11_VERTEX_BUFFER;
class MATERIAL;

// FONT
//   Texture-base font, that uses a pre-generated texture, in which a set of all required characters 
//   are stored. All information (used material, font-texture, font parameters) are stored in a simple
//   custom file-format (".font").
//   When outputting text, every character is looked up in the font-texture by texture coordinates and 
//   the corresponding part of font-texture is mapped onto a quad. All text from the same font is 
//   collected and rendered in a single draw-batch.
class FONT
{        
public:	
	FONT()
	{
		active = true;
		textureWidth = 0;
	  textureHeight = 0;	 
	  fontHeight = 0;
	  fontSpacing = 0;
		numTexCoords = 0;
	  texCoords = NULL;
		material = NULL;
		vertexBuffer = NULL;
		rtConfig = NULL;
	}

	~FONT()
	{
		Release();
	}

	void Release();

	bool Load(const char *fileName);	

	void Print(const VECTOR2D &position,float scale,const COLOR &color,const char *string,...);

	void AddSurfaces();  

	void SetActive(bool active)
	{
		this->active = active;
	}    

	bool IsActive() const
	{
		return active;
	}

	const char* GetName() const
	{ 
		return name;
	}

private:
	bool active;
	char name[DEMO_MAX_FILENAME];

	int textureWidth,textureHeight;	// width/ height of font-texture
	int fontHeight; // height of used font
	int fontSpacing; // spacing of used font
	int numTexCoords; // number of texCoords
	float *texCoords; // texCoords for fetching corresponding part for each character from font-texture
  
	MATERIAL *material;
	DX11_VERTEX_BUFFER *vertexBuffer;
	RENDER_TARGET_CONFIG *rtConfig;

};

#endif
