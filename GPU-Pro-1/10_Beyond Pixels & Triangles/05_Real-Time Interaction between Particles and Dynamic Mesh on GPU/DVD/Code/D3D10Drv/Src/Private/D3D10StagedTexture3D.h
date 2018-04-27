#ifndef D3D10DRV_D3D10STAGEDTEXTURE3D_H_INCLUDED
#define D3D10DRV_D3D10STAGEDTEXTURE3D_H_INCLUDED

#include "Forw.h"

#include "D3D10StagedResourceImpl.h"

namespace Mod
{

	class D3D10StagedTexture3D : public D3D10StagedResourceImpl< ID3D10Texture3D >
	{
		// types
	public:

		// constructors / destructors
	public:
		explicit D3D10StagedTexture3D( const StagedResourceConfig& cfg, ID3D10Device* dev, UINT64 resSize );
		~D3D10StagedTexture3D();
	
		// manipulation/ access
	public:

		// data
	private:

	};
}

#endif