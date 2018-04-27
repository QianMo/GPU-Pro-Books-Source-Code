#ifndef D3D9DRV_D3D9INPUTLAYOUT_H_INCLUDED
#define D3D9DRV_D3D9INPUTLAYOUT_H_INCLUDED

#include "Wrap3D/Src/InputLayout.h"

#include "Forw.h"

namespace Mod
{

	class D3D9InputLayout : public InputLayout
	{
		// types & constants
	public:
		static const UINT VERTEX_DATA = UINT(-1);
		typedef Types<UINT> :: Vec UIntVec;

		// construction/ destruction
	public:
		explicit D3D9InputLayout( const InputLayoutConfig& cfg, const D3D9CapsConstants& consts, IDirect3DDevice9* dev );
		~D3D9InputLayout();

	public:
		void Bind( IDirect3DDevice9* dev ) const;
		void SetSSFreqs( IDirect3DDevice9* dev, UINT32 numInstances ) const;
		void RestoreSSFreqs( IDirect3DDevice9* dev ) const;

		// data
	private:
		ComPtr< IDirect3DVertexDeclaration9 >	mResource;
		UIntVec									mStreamSourceFreqs;
	};

}

#endif