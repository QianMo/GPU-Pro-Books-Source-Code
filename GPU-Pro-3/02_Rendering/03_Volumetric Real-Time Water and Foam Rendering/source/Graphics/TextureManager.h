#ifndef __TEXTUREMANAGER__H__
#define __TEXTUREMANAGER__H__

#include <string>
#include <map>

#include "../Util/Singleton.h"

class TextureManager : public Singleton<TextureManager>
{
	friend class Singleton<TextureManager>;

public:
	enum FilterMethod
	{
		FILTER_NONE = 0,
		FILTER_NEAREST,
		FILTER_LINEAR
	};
	TextureManager(void);
	~TextureManager(void);

	/// Loads a texture from file
	unsigned int LoadTexture(const char* filename, bool mipMaps=true, bool compressed=true, bool isSkyBoxTexture=false);

	/// Loads a cube map from file
	unsigned int LoadCubeMap(const char* filename[6], bool mipMaps);

	/// Releases a texture
	void ReleaseTexture(const unsigned int& idx);

	/// Deletes all textures
	void DeleteAllTextures(void);

	/// Returns the size of a texture
	unsigned int GetTextureSize(const unsigned int& idx);

	/// Returns the size of all loaded textures
	unsigned int GetTotalTextureSize(void);

	/// Changes the filtermethod
	void ChangeFilterMethod(const FilterMethod& method);

	/// Changes the mipmapmethod
	void ChangeMipMapMethod(const FilterMethod& method);

	/// Returns the current filtermethod
	const FilterMethod GetFilterMethod(void) const { return (FilterMethod)filterMethod; }

	/// Returns the current mipmapmethod
	const FilterMethod GetMipMapMethod(void) const { return (FilterMethod)mipMapMethod; }

private:
	struct TextureReference {
		unsigned int idx;
		unsigned int ref;
		unsigned int size;
		unsigned int numMipMaps;

		TextureReference(unsigned id_, unsigned sz, unsigned mm) : idx(id_), ref(1), size(sz), numMipMaps(mm) {}
		TextureReference() : idx(0), ref(0), size(0), numMipMaps(0) {}
		TextureReference(const TextureReference& tr) : idx(tr.idx), ref(tr.ref), size(tr.size), numMipMaps(tr.numMipMaps) {}
	};

	/// Creates the open gl texture
	unsigned int CreateTexture(const unsigned int& Format, unsigned char** pixels, const unsigned int& width, const unsigned int& height, const unsigned int& numMipMaps, const bool& compressed, const bool& isSkyBoxTexture);

	unsigned int CreateCubeMap(unsigned char** pixels[6], const unsigned int& width, const unsigned int& height, const unsigned int& numMipMaps);

	/// Textures
	std::map<std::string, TextureReference> textures;
	std::map<std::string, TextureReference>::iterator iter;

	/// Current filtermethod
	unsigned int filterMethod;

	/// Current mipmapmethod
	unsigned int mipMapMethod;
};

#endif