#include "Precompiled.h"

#include "D3D9CapsConstants.h"

namespace Mod
{
	void FillD3D9CapsConstants( D3D9CapsConstants& caps, IDirect3D9* d3d9, UINT adapterID, D3DDEVTYPE devTypes )
	{
		D3DCAPS9 d3dcaps;
		MD_D3DV( d3d9->GetDeviceCaps( adapterID, devTypes, &d3dcaps ) );
		
		caps.NUM_TEXTURE_SLOTS		= d3dcaps.MaxSimultaneousTextures;
		caps.NUM_VERTEXBUFFER_SLOTS	= d3dcaps.MaxStreams;
		caps.MAX_TEXTURE_DIMMENSION	= std::min( d3dcaps.MaxTextureWidth, d3dcaps.MaxTextureHeight );
	}
}