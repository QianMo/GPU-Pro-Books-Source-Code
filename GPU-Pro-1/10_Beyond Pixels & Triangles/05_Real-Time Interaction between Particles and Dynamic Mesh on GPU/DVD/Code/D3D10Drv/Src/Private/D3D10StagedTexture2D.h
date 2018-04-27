#ifndef D3D10DRV_D3D10STAGEDTEXTURE2D_H_INCLUDED
#define D3D10DRV_D3D10STAGEDTEXTURE2D_H_INCLUDED

#include "Forw.h"

#include "D3D10StagedResourceImpl.h"

namespace Mod
{

	class D3D10StagedTexture2D : public D3D10StagedResourceImpl< ID3D10Texture2D >
	{
		// types
	public:

		// constructors / destructors
	public:
		explicit D3D10StagedTexture2D( const StagedResourceConfig& cfg, ID3D10Device* dev, UINT64 resSize );
		~D3D10StagedTexture2D();
	
		// manipulation/ access
	public:

		// data
	private:

	};
}

#endif