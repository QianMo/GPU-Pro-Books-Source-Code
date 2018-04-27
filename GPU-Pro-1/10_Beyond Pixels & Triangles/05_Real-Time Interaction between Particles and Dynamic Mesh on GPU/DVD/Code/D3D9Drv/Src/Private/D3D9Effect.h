#ifndef D3D9DRV_D3D9EFFECT_H_INCLUDED
#define D3D9DRV_D3D9EFFECT_H_INCLUDED

#include "Wrap3D\Src\Effect.h"

namespace Mod
{

	class D3D9Effect : public Effect
	{
		// types
	public:
		typedef ComPtr<ID3DXEffect> ResourcePtr;
		typedef Effect Base;

		struct Technique
		{
			D3DXHANDLE techHandle;
			UINT32 numPasses;
		};

		// construction/ destrcution
	public:
		D3D9Effect( const EffectConfig& cfg, IDirect3DDevice9* dev );
		virtual ~D3D9Effect();

		// manipulation/ access
	public:
		void SetStateManager( ID3DXEffectStateManager* manager );

	private:
		D3D9Effect( const EffectConfig& cfg, ResourcePtr eff );

		// polymorphism
	private:
		virtual void				SetTechniqueImpl( const AnsiString& name ) OVERRIDE;
		virtual bool				BindImpl( UINT32 passNum ) OVERRIDE;
		virtual EffectVarBindPtr	GetVarBindImpl( const CodeString& name ) OVERRIDE;
		virtual UINT32				GetNumPassesImpl() const;

		// data
	private:
		ResourcePtr		mResource;
		Technique		mCurrentTechnique;
	
	};

}

#endif