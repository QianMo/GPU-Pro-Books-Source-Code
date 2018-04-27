#ifndef D3D10DRV_D3D10BUFFER_INDEX_H_INCLUDED
#define D3D10DRV_D3D10BUFFER_INDEX_H_INCLUDED

#include "D3D10BufferImpl.h"

namespace Mod
{

	struct D3D10Buffer_IndexConfig
	{
		typedef class D3D10Buffer_Index		Child;
		typedef IndexBufferConfig			BufConfigType;
	};

	class D3D10Buffer_Index : public D3D10BufferImpl< D3D10Buffer_IndexConfig >
	{
		// construction/ destruction
	public:
		D3D10Buffer_Index( const BufConfigType& cfg, ID3D10Device* dev );
		virtual ~D3D10Buffer_Index();

		// manipulation/ access
	public:
		void BindTo( ID3D10Device * dev ) const;

		static void SetBindToZero( ID3D10Device * dev );

	};
}

#endif