#include "Precompiled.h"

#include "Math/Src/Operations.h"

#include "Wrap3D/Src/InputLayoutConfig.h"
#include "Wrap3D/Src/Format.h"

#include "RawVertexData.h"
#include "BlendIndicesVCFactory.h"


namespace Mod
{
	BlendIndicesVCFactory::BlendIndicesVCFactory( const VertexCmptFactoryConfig& cfg ) : 
	Base( cfg )
	{
	}

	//------------------------------------------------------------------------

	BlendIndicesVCFactory::~BlendIndicesVCFactory()
	{

	}

	//------------------------------------------------------------------------

	void
	BlendIndicesVCFactory::ModifyILElemConfigImpl( ILElement& el ) const
	{
		el.semantics = "BLENDINDICES";
	}

	//------------------------------------------------------------------------

	void
	BlendIndicesVCFactory::ConstructVertexDataImpl( const RawVertexData& rawData,  UINT32 idx, UINT32 disp, const Format* fmt, Bytes& oData ) const
	{
		using namespace Math;

		const RawVertexData::BlendIndices& sourceComp = rawData.blendIndices[ idx ];

		size_t vertSize = size_t(oData.GetSize() / sourceComp.size());

		for( size_t i = 0, p = disp, e = sourceComp.size(); i < e; i++, p+= vertSize)
		{
			fmt->Convert( sourceComp[i].elems, &oData[p] );
		}
	}

}

