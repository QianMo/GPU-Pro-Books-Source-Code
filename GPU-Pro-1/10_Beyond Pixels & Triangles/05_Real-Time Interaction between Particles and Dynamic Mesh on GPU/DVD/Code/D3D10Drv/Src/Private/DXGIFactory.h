#ifndef D3D10DRV_DXGIFACTORY_H_INCLUDED
#define D3D10DRV_DXGIFACTORY_H_INCLUDED

#include "Forw.h"

namespace Mod
{

	class DXGIFactory
	{
		// types
	public:
		typedef ComPtr< IDXGIFactory > Ptr;
		typedef Types< DXGI_MODE_DESC > :: Vec ModeDescs;

		struct SimplifiedModeDesc
		{
			UINT32 width;
			UINT32 height;
		};

		typedef Types< SimplifiedModeDesc > :: Vec SimplifiedModeDescs;

		// constructors / destructors
	public:
		explicit DXGIFactory();
		~DXGIFactory();
	
		// manipulation/ access
	public:
		static DXGIFactory& Single();

		const Ptr& Get() const;
		const ModeDescs& GetModeDescs() const;
		const SimplifiedModeDescs& GetSimplifiedModeDescs() const;

		// because the release of the last IDXGIFactory cannot be called form DLLMain
		void Release();

		// data
	private:
		Ptr								mDXGIFactory;
		ModeDescs						mModeDescs;
		SimplifiedModeDescs				mSimplifiedModeDescs;

#ifndef MD_D3D10_STATIC_LINK
		DynamicLibraryPtr				mDXGILibrary;
#endif

	};
}

#endif