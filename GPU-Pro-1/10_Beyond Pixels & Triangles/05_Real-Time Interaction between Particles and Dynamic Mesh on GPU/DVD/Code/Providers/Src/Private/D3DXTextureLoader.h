#ifndef PROVIDERS_D3DXTEXTURELOADER_H_INCLUDED
#define PROVIDERS_D3DXTEXTURELOADER_H_INCLUDED

#include "Forw.h"

#include "D3D9Helpers/Src/D3D9FormatMap.h"

#include "TextureLoader.h"

namespace Mod
{

	class D3DXTextureLoader : public TextureLoader
	{
		// types
	public:

		// constructors / destructors
	public:
		explicit D3DXTextureLoader( const TextureLoaderConfig& cfg );
		~D3DXTextureLoader();
	
		// manipulation/ access
	public:

		// polymorphism
	private:
		virtual TextureConfigPtr LoadImpl( const Bytes& data, const String& fileExtension ) OVERRIDE;

		// data
	private:
		IDirect3D9*			mD3D9;
		IDirect3DDevice9*	mD3D9Device;

		MemPtr< D3D9FormatMap >	mD3D9FormatMap;
	};
}

#endif