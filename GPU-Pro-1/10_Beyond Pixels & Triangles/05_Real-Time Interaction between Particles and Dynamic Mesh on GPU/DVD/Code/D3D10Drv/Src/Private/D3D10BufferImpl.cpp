#include "Precompiled.h"

#include "Wrap3D\Src\BufferConfig.h"

#include "D3D10BufferCommon.h"

#include "D3D10BufferImpl.h"

#include "D3D10Buffer_Vertex.h"
#include "D3D10Buffer_Index.h"
#include "D3D10Buffer_Shader.h"
#include "D3D10Buffer_ShaderIndex.h"
#include "D3D10Buffer_ShaderVertex.h"
#include "D3D10Buffer_SOVertex.h"
#include "D3D10Buffer_SOShader.h"
#include "D3D10Buffer_SOShaderVertex.h"
#include "D3D10Buffer_Constant.h"

namespace Mod
{
	template <typename Config>
	D3D10BufferImpl<Config>::D3D10BufferImpl( const BufConfigType& cfg, UINT32 bindFlags, ID3D10Device* dev ) :
	Base( cfg, AddExtraFlags( cfg, bindFlags ), dev )
	{
	}

	//------------------------------------------------------------------------

	template <typename Config>
	D3D10BufferImpl<Config>::~D3D10BufferImpl()
	{
	}

	//------------------------------------------------------------------------

	template class D3D10BufferImpl< D3D10Buffer_Vertex::Config			>;
	template class D3D10BufferImpl< D3D10Buffer_Index::Config			>;
	template class D3D10BufferImpl< D3D10Buffer_Shader::Config			>;
	template class D3D10BufferImpl< D3D10Buffer_ShaderIndex::Config		>;
	template class D3D10BufferImpl< D3D10Buffer_ShaderVertex::Config	>;
	template class D3D10BufferImpl< D3D10Buffer_SOVertex::Config		>;
	template class D3D10BufferImpl< D3D10Buffer_SOShader::Config		>;
	template class D3D10BufferImpl< D3D10Buffer_SOShaderVertex::Config	>;
	template class D3D10BufferImpl< D3D10Buffer_Constant::Config		>;
}