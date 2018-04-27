#ifndef D3D10DRV_D3D10BUFFER_SOSHADER_H_INCLUDED
#define D3D10DRV_D3D10BUFFER_SOSHADER_H_INCLUDED

#include "D3D10SOBufferImpl.h"

namespace Mod
{

	struct D3D10Buffer_SOShaderConfig
	{
		typedef class D3D10Buffer_SOShader		Child;
		typedef SOShaderBufferConfig			BufConfigType;
	};

	class D3D10Buffer_SOShader : public D3D10SOBufferImpl< D3D10Buffer_SOShaderConfig >
	{
	public:
		D3D10Buffer_SOShader( const BufConfigType& cfg, ID3D10Device* dev );
		virtual ~D3D10Buffer_SOShader();

	};

}

#endif