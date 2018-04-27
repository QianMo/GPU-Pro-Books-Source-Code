#ifndef D3D10DRV_D3D10BUFFER_SHADER_H_INCLUDED
#define D3D10DRV_D3D10BUFFER_SHADER_H_INCLUDED

#include "D3D10BufferImpl.h"

namespace Mod
{

	struct D3D10Buffer_ShaderConfig
	{
		typedef class D3D10Buffer_Shader	Child;
		typedef ShaderBufferConfig			BufConfigType;
	};

	class D3D10Buffer_Shader : public D3D10BufferImpl< D3D10Buffer_ShaderConfig >
	{
	public:
		D3D10Buffer_Shader( const BufConfigType& cfg, ID3D10Device* dev );
		virtual ~D3D10Buffer_Shader();

	};
}

#endif