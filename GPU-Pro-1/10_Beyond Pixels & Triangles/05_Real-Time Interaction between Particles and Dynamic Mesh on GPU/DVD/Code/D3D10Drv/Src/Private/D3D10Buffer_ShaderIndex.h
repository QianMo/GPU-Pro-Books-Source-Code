#ifndef D3D10DRV_D3D10BUFFER_SHADERINDEX_H_INCLUDED
#define D3D10DRV_D3D10BUFFER_SHADERINDEX_H_INCLUDED

#include "D3D10BufferImpl.h"

namespace Mod
{

	struct D3D10Buffer_ShaderIndexConfig
	{
		typedef class D3D10Buffer_ShaderIndex	Child;
		typedef ShaderIndexBufferConfig			BufConfigType;
	};

	class D3D10Buffer_ShaderIndex : public D3D10BufferImpl< D3D10Buffer_ShaderIndexConfig >
	{
	public:
		D3D10Buffer_ShaderIndex( const BufConfigType& cfg, ID3D10Device* dev );
		virtual ~D3D10Buffer_ShaderIndex();

	};

}

#endif