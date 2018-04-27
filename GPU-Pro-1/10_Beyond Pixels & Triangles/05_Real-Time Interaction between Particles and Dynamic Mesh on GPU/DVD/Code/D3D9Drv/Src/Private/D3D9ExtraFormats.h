#ifndef D3D9DRV_D3D9EXTRAFORMATS_H_INCLUDED
#define D3D9DRV_D3D9EXTRAFORMATS_H_INCLUDED

namespace Mod
{
	enum D3D9ExtraFormats
	{
		MDFMT_R32G32B32_FLOAT = 2000,
	};

	bool IsFormatExtra( UINT32 fmt );

}

#endif