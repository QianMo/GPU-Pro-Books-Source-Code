#ifndef D3D9DRV_D3D9USAGE_H_INCLUDED
#define D3D9DRV_D3D9USAGE_H_INCLUDED

#include "Wrap3D\Src\Usage.h"

namespace Mod
{

	struct D3D9UsageConfig
	{
		DWORD			textureUsage;
		D3DPOOL 		texPool;
		DWORD			bufferUsage;
		D3DPOOL			bufPool;
		DWORD			lockFlags;
	};

	class D3D9Usage : public Usage
	{
		// construction/ destruction
	public:
		D3D9Usage( const D3D9UsageConfig& cfg );
		virtual ~D3D9Usage();

		// manipulation/ access
	public:
		const D3D9UsageConfig& GetConfig() const;

		// data
	private:
		D3D9UsageConfig mConfig;
	};

}


#endif