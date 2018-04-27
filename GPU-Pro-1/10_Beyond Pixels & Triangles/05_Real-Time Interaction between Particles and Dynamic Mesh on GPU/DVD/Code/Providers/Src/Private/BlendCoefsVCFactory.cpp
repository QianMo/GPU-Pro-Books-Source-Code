#include "Precompiled.h"

#include "Math/Src/Operations.h"

#include "Wrap3D/Src/InputLayoutConfig.h"
#include "Wrap3D/Src/Format.h"

#include "RawVertexData.h"
#include "BlendCoefsVCFactory.h"


namespace Mod
{
	BlendCoefsVCFactory::BlendCoefsVCFactory( const VertexCmptFactoryConfig& cfg ) : 
	Base( cfg )
	{
	}

	//------------------------------------------------------------------------

	BlendCoefsVCFactory::~BlendCoefsVCFactory()
	{

	}

	//------------------------------------------------------------------------

	void
	BlendCoefsVCFactory::ModifyILElemConfigImpl( ILElement& el ) const
	{
		el.semantics = "BLENDWEIGHT";
	}

	//------------------------------------------------------------------------

	void
	BlendCoefsVCFactory::ConstructVertexDataImpl( const RawVertexData& rawData,  UINT32 idx, UINT32 disp, const Format* fmt, Bytes& oData ) const
	{
		using namespace Math;

		const RawVertexData::Float4Vec& sourceComp = rawData.blendCoefs[ idx ];

		size_t vertSize = size_t(oData.GetSize() / sourceComp.size());

		for( size_t i = 0, p = disp, e = sourceComp.size(); i < e; i++, p+= vertSize)
		{
			fmt->Convert( sourceComp[i].elems, &oData[p] );
		}
	}

}

