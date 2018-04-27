#include "Precompiled.h"
#include "Wrap3D/Src/InputLayoutConfig.h"

#include "D3D10Format.h"
#include "D3D10InputLayout.h"
#include "D3D10Effect.h"
#include "D3D10Exception.h"
#include "D3D10Device.h"

namespace Mod
{
	D3D10InputLayout::D3D10InputLayout( const InputLayoutConfig& cfg, ID3D10Device* dev ) : 
	Parent( cfg )
	{

		Types< D3D10_INPUT_ELEMENT_DESC > :: Vec inputElements( cfg.elements.size() );
		
		UINT32 prevAccumulatedSizes[D3D10Device::NUM_VERTEXBUFFER_SLOTS] = {};

		for( size_t i = 0, n = inputElements.size(); i < n; ++i )
		{
			D3D10_INPUT_ELEMENT_DESC& e = inputElements[i];
			const ILElement& src = cfg.elements[i];

			MD_FERROR_ON_FALSE( src.fmt );

			const D3D10Format* fmt = static_cast<const D3D10Format*>(src.fmt);

			prevAccumulatedSizes[src.inputSlot] += src.numBytesToSkip;

			e.SemanticName			= src.semantics.c_str();
			e.SemanticIndex			= src.semIndex;
			e.Format				= fmt->GetValue();
			e.InputSlot				= src.inputSlot;
			e.AlignedByteOffset		= prevAccumulatedSizes[src.inputSlot];
			e.InputSlotClass		= src.instanceDataStepRate ? D3D10_INPUT_PER_INSTANCE_DATA : D3D10_INPUT_PER_VERTEX_DATA;
			e.InstanceDataStepRate	= src.instanceDataStepRate;

			prevAccumulatedSizes[src.inputSlot] += GetByteCountChecked( fmt );
		}

		UINT ieNum = UINT(inputElements.size());

		if( inputElements.empty() )
		{
			inputElements.resize( 1 );
			D3D10_INPUT_ELEMENT_DESC& e = inputElements.back();

			e.SemanticName			= "";
			e.SemanticIndex			= 0;
			e.Format				= DXGI_FORMAT_UNKNOWN;
			e.InputSlot				= 0;
			e.AlignedByteOffset		= 0;
			e.InputSlotClass		= D3D10_INPUT_PER_VERTEX_DATA;
			e.InstanceDataStepRate	= 0;
		}

		ID3D10InputLayout* resultingIL;
		const void *inputSignature;
		UINT32 byteCodeLength;

		static_cast<const D3D10Effect*>(&*cfg.associatedEffect)->FillInputSignatureParams( inputSignature, byteCodeLength );	

		D3D10_THROW_IF( dev->CreateInputLayout(	&inputElements[0], ieNum, 
												inputSignature, static_cast<SIZE_T>(byteCodeLength),
												&resultingIL ) );

		mResource.set( resultingIL );

	}

	//------------------------------------------------------------------------

	D3D10InputLayout::~D3D10InputLayout()
	{
	}

	//------------------------------------------------------------------------

	void
	D3D10InputLayout::Bind( ID3D10Device* dev ) const
	{
		dev->IASetInputLayout( &*mResource );
	}


}