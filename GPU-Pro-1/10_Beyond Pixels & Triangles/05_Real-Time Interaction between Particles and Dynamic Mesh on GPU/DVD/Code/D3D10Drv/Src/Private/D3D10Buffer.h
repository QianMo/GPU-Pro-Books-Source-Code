#ifndef D3D10DRV_D3D10BUFFER_H_INCLUDED
#define D3D10DRV_D3D10BUFFER_H_INCLUDED

#include "Wrap3D\Src\Buffer.h"

namespace Mod
{
	class D3D10Buffer : public Buffer
	{
		// types
	public:
		typedef Parent Base;
		typedef D3D10Buffer Parent;

		typedef D3D10_BUFFER_DESC DescType;
		typedef ID3D10Buffer ResType;
		typedef ComPtr<ResType> ResourcePtr;

		struct IABindSlot
		{

			bool operator ! () const;

			ID3D10Buffer**	buffer;
			UINT*			stride;
			UINT*			offset;
		};

		struct SOBindSlot
		{
			SOBindSlot();

			bool operator ! () const;

			ID3D10Buffer**	buffer;
			UINT*			offset;
			bool			set;
		};

	public:
		explicit D3D10Buffer( const BufferConfig& cfg, UINT32 bindFlags, ID3D10Device* dev );
		virtual ~D3D10Buffer() = 0;

		// manipulation/ access
	public:
		const ResourcePtr&		GetResource() const;
		UINT32					GetBindFlags() const;

		void					BindTo( IABindSlot& slot ) const;
		static void				SetBindToZero( IABindSlot& slot );

		void					BindTo( SOBindSlot& slot ) const;
		static void				SetBindToZero( SOBindSlot& slot );

		void					BindTo( ID3D10EffectConstantBuffer* slot ) const;
		static void				SetBindToZero( ID3D10EffectConstantBuffer* slot );

		// child manipulation
	protected:
		void					SetResource( ResourcePtr::PtrType res );
		ResourcePtr::PtrType	GetResourceInternal() const;

		// polymorphism
	private:
		virtual void BindToImpl( IABindSlot& slot ) const;
		virtual void BindToImpl( SOBindSlot& slot ) const;
		virtual void BindToImpl( ID3D10EffectConstantBuffer* slot ) const;

		virtual void MapImpl( void ** ptr, MapType type ) OVERRIDE;
		virtual void UnmapImpl() OVERRIDE;

		// data
	private:
		ResourcePtr	mResource;
		UINT32		mBindFlags;
	};
}

#endif