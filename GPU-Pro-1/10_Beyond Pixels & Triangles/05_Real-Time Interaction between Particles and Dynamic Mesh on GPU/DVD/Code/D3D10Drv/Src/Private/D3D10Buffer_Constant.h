#ifndef D3D10DRV_D3D10BUFFER_CONSTANT_H_INCLUDED
#define D3D10DRV_D3D10BUFFER_CONSTANT_H_INCLUDED

#include "D3D10VertexBufferImpl.h"

namespace Mod
{

	struct D3D10Buffer_ConstantConfig
	{
		typedef class D3D10Buffer_Constant	Child;
		typedef ConstantBufferConfig		BufConfigType;
	};

	class D3D10Buffer_Constant : public D3D10BufferImpl< D3D10Buffer_ConstantConfig >
	{
		// construction/ destruction
	public:
		D3D10Buffer_Constant( const BufConfigType& cfg, ID3D10Device* dev );
		virtual ~D3D10Buffer_Constant();

		// polymorphism
	private:
		virtual void BindToImpl( ID3D10EffectConstantBuffer* slot ) const OVERRIDE;

	};
}
#endif