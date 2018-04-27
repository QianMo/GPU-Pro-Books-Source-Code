#ifndef D3D10DERV_D3D10STAGEDRESOURCEIMPL_H_INCLUDED
#define D3D10DERV_D3D10STAGEDRESOURCEIMPL_H_INCLUDED

#include "Forw.h"

#include "Wrap3D/Src/StagedResource.h"

namespace Mod
{
	template < typename R >
	class D3D10StagedResourceImpl : public StagedResource
	{
		// types
	public:
		typedef ComPtr< R >				ResourcePtr;
		typedef D3D10StagedResourceImpl	Parent;
		typedef StagedResource			Base;

		// construction/ destruction
	public:
		D3D10StagedResourceImpl( const StagedResourceConfig& cfg, ID3D10Device *dev, UINT64 resourceSize );
		~D3D10StagedResourceImpl();

		// manipulation/ access
	public:
		void Sync( ID3D10Device * dev );

		// polymorphism
	private:
		virtual UINT64	GetSizeImpl() const OVERRIDE;
		virtual bool	GetDataImpl( Bytes& oBytes ) OVERRIDE;

		// data
	private:
		ResourcePtr mResource;
		UINT64		mResourceSize;
	};
}

#endif