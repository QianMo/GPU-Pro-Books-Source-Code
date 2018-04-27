#ifndef D3D9DRV_D3D9BUFFER_H_INCLUDED
#define D3D9DRV_D3D9BUFFER_H_INCLUDED

#include "Wrap3D\Src\Buffer.h"

namespace Mod
{
	class D3D9Buffer : public Buffer
	{
		// types
	public:
		typedef ComPtr<IDirect3DResource9> ResourcePtr;
		typedef Parent		Base;
		typedef D3D9Buffer	Parent;

	public:
		explicit D3D9Buffer( const BufferConfig& cfg );
		virtual ~D3D9Buffer() = 0;

		// manipulation/ access
	public:
		const ResourcePtr&		GetResource() const;
		void					BindAsVB( IDirect3DDevice9* dev, UINT32 slot ) const;
		void					BindAsIB( IDirect3DDevice9* dev ) const;

		// child manipulation
	protected:
		void					SetResource( ResourcePtr::PtrType res );
		ResourcePtr::PtrType	GetResourceInternal() const;

		// polymorphism
	private:
		virtual void			BindAsVBImpl( IDirect3DDevice9* dev, UINT32 slot ) const;
		virtual void			BindAsIBImpl( IDirect3DDevice9* dev ) const;

		// data
	private:
		ResourcePtr	mResource;
	};
}

#endif