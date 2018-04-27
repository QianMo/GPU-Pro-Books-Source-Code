#include "Precompiled.h"
#include "Wrap3D/Src/InputLayoutConfig.h"

#include "D3D9Format.h"
#include "D3D9InputLayout.h"
#include "D3D9Effect.h"
#include "D3D9Device.h"

#include "D3D9Exception.h"

#include "D3D9SemanticsMap.h"

namespace Mod
{
	D3D9InputLayout::D3D9InputLayout( const InputLayoutConfig& cfg, const D3D9CapsConstants& consts, IDirect3DDevice9* dev ) : 
	Parent( cfg )
	{

		Types< D3DVERTEXELEMENT9 > :: Vec inputElements( cfg.elements.size() + 1 );
		typedef Types< UINT32 > :: Vec UIntVec;
		UIntVec prevAccumulatedSizes( consts.NUM_VERTEXBUFFER_SLOTS, 0 );

		mStreamSourceFreqs.resize( inputElements.size(), VERTEX_DATA );

		// D3D9 end marker
		{
			D3DVERTEXELEMENT9 end = D3DDECL_END();
			inputElements.back() = end;
		}

		for( size_t i = 0, n = inputElements.size()-1; i < n; ++i )
		{
			D3DVERTEXELEMENT9& e = inputElements[i];
			const ILElement& src = cfg.elements[i];

			MD_FERROR_ON_FALSE( src.fmt );

			const D3D9Format* fmt = static_cast<const D3D9Format*>(src.fmt);

			prevAccumulatedSizes[src.inputSlot] += src.numBytesToSkip;

			e.Stream				= (WORD)src.inputSlot;											// Stream index
			e.Offset				= (WORD)prevAccumulatedSizes[src.inputSlot];					// Offset in the stream in bytes
			e.Type					= (BYTE)fmt->GetVertexDeclType();								// Data type
			e.Method				= D3DDECLMETHOD_DEFAULT;										// Processing method
			e.Usage					= (BYTE)D3D9SemanticsMap::Single().GetUsage( src.semantics );	// Semantics
			e.UsageIndex			= (BYTE)src.semIndex;													// Semantic index

			if( src.instanceDataStepRate )
			{
				mStreamSourceFreqs[i]	= src.instanceDataStepRate;
			}
			
			prevAccumulatedSizes[src.inputSlot] += GetByteCountChecked( fmt );
		}

		// in case no elements are on input, we emulate DX10 style SV_VertexID by binding a float buffer that goes {0.f,1.f,2.f...}
		if( inputElements.size() == 1 )
		{
			inputElements.resize( inputElements.size() + 1, inputElements.back() );
			D3DVERTEXELEMENT9& e = inputElements.back();

			e.Stream				= 0;															// Stream index
			e.Offset				= 0;															// Offset in the stream in bytes
			e.Type					= D3DDECLTYPE_FLOAT1;											// Data type
			e.Method				= D3DDECLMETHOD_DEFAULT;										// Processing method
			e.Usage					= (BYTE)D3D9SemanticsMap::Single().GetUsage( "TEXTURE" );		// Semantics
			e.UsageIndex			= D3DDECLUSAGE_TEXCOORD ;										// Semantic index
		}

		IDirect3DVertexDeclaration9 * vertexDecl(NULL);

		D3D9_THROW_IF( dev->CreateVertexDeclaration( &inputElements[0],	&vertexDecl ) );

		mResource.set( vertexDecl );

	}

	//------------------------------------------------------------------------

	D3D9InputLayout::~D3D9InputLayout()
	{
	}

	//------------------------------------------------------------------------

	void
	D3D9InputLayout::Bind( IDirect3DDevice9* dev ) const
	{
		MD_D3DV( dev->SetVertexDeclaration( &*mResource ) );
	}

	//------------------------------------------------------------------------

	void
	D3D9InputLayout::SetSSFreqs( IDirect3DDevice9* dev, UINT32 numInstances ) const
	{
		for( UINT32 i = 0, e = (UINT32)mStreamSourceFreqs.size(); i < e; i ++ )
		{
			if( mStreamSourceFreqs[ i ] == VERTEX_DATA )
			{
				MD_D3DV( dev->SetStreamSourceFreq( i, D3DSTREAMSOURCE_INDEXEDDATA | numInstances ) );
			}
			else
			{
				MD_D3DV( dev->SetStreamSourceFreq( i, D3DSTREAMSOURCE_INSTANCEDATA | mStreamSourceFreqs[ i ] ) );
			}
		}
	}

	//------------------------------------------------------------------------

	void
	D3D9InputLayout::RestoreSSFreqs( IDirect3DDevice9* dev ) const
	{
		for( UINT32 i = 0, e = (UINT32)mStreamSourceFreqs.size(); i < e; i ++ )
		{
			MD_D3DV( dev->SetStreamSourceFreq( i, 1 ) );
		}
	}


}