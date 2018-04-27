#include "Precompiled.h"

#include "D3D10StagedTexture3D.h"

namespace Mod
{
	D3D10StagedTexture3D::D3D10StagedTexture3D( const StagedResourceConfig& cfg, ID3D10Device* dev, UINT64 resSize ) :
	Parent( cfg, dev, resSize )
	{
	}

	//------------------------------------------------------------------------

	D3D10StagedTexture3D::~D3D10StagedTexture3D() 
	{
	}
}