#ifndef D3D10DRV_D3D10USAGEMAP_H_INCLUDED
#define D3D10DRV_D3D10USAGEMAP_H_INCLUDED

namespace Mod
{
	class D3D10UsageMap
	{
		// types
	public:
		typedef Types2< D3D10_USAGE, const class Usage* >::Map Map;

		// construction/ destruction
	public:
		explicit D3D10UsageMap( const class Usages& Usage );
		~D3D10UsageMap();

	public:
		const Usage* GetUsage( D3D10_USAGE usg ) const;

	private:
		Map mMap;

	};
}

#endif