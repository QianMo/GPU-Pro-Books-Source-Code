#ifndef D3D10DRV_D3D10USAGE_H_INCLUDED
#define D3D10DRV_D3D10USAGE_H_INCLUDED

#include "Wrap3D\Src\Usage.h"

namespace Mod
{

	class D3D10Usage : public Usage
	{
		// construction/ destruction
	public:
		D3D10Usage( D3D10_USAGE usage, INT32 defCPUAccessFlags );
		virtual ~D3D10Usage();

		// manipulation/ access
	public:
		D3D10_USAGE	GetValue() const;
		INT32		GetDefaultAccessFlags() const;

		// data
	private:
		D3D10_USAGE				mD3D10Usage;
		INT32					mDefaultCPUAccessFlags;
	};

}

#endif