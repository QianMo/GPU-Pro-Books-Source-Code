#ifndef D3D10DRV_D3D10STAGEDBUFFER_H_INCLUDED
#define D3D10DRV_D3D10STAGEDBUFFER_H_INCLUDED

#include "Forw.h"

#include "D3D10StagedResourceImpl.h"

namespace Mod
{

	class D3D10StagedBuffer : public D3D10StagedResourceImpl< ID3D10Buffer >
	{
		// types
	public:

		// constructors / destructors
	public:
		explicit D3D10StagedBuffer( const StagedResourceConfig& cfg, ID3D10Device* dev, UINT64 resSize );
		~D3D10StagedBuffer();
	
		// manipulation/ access
	public:

		// data
	private:

	};
}

#endif