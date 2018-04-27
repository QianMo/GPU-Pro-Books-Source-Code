#include "Precompiled.h"

#include "D3D10StagedBuffer.h"

namespace Mod
{
	D3D10StagedBuffer::D3D10StagedBuffer( const StagedResourceConfig& cfg, ID3D10Device* dev, UINT64 resSize ) :
	Parent( cfg, dev, resSize )
	{
	}

	//------------------------------------------------------------------------

	D3D10StagedBuffer::~D3D10StagedBuffer() 
	{
	}
}