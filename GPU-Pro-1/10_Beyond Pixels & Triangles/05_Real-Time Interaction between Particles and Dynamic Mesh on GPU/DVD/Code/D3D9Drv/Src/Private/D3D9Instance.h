#ifndef D3D9DRV_D3D9INSTANCE_H_INCLUDED
#define D3D9DRV_D3D9INSTANCE_H_INCLUDED

#include "Forw.h"

namespace Mod
{

	class D3D9Instance
	{
		// types
	public:
		typedef ComPtr< IDirect3D9 > Ptr;
		typedef Types< D3DDISPLAYMODE > :: Vec ModeDescs;

		struct SimplifiedModeDesc
		{
			UINT32 width;
			UINT32 height;
		};

		typedef Types< SimplifiedModeDesc > :: Vec SimplifiedModeDescs;

		// constructors / destructors
	public:
		explicit D3D9Instance();
		~D3D9Instance();
	
		// manipulation/ access
	public:
		static D3D9Instance& Single();
		static bool& Exists();

		const Ptr& Get() const;
		const ModeDescs& GetModeDescs() const;
		const SimplifiedModeDescs& GetSimplifiedModeDescs() const;

		// because the release of the last ID3D9Instance cannot be called form DLLMain
		void Release();

		// data
	private:
		Ptr								mD3D9Instance;
		ModeDescs						mModeDescs;
		SimplifiedModeDescs				mSimplifiedModeDescs;
	};
}

#endif