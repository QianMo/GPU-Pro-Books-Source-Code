#ifndef D3D10DRV_D3D10EFFECT_H_INCLUDED
#define D3D10DRV_D3D10EFFECT_H_INCLUDED

#include "Wrap3D\Src\Effect.h"

namespace Mod
{

	class D3D10Effect : public Effect
	{
		friend class D3D10EffectPool;

		// types
	public:
		typedef ComPtr<ID3D10Effect> ResourcePtr;

		struct Technique
		{
			UINT32 numPasses;
			ID3D10EffectTechnique* tech;

			Technique();
			~Technique();

			ID3D10EffectTechnique* operator ->() const;
		};		
		
		// construction/ destrcution
	public:
		D3D10Effect( const EffectConfig& cfg, ID3D10Device* dev );		
		virtual ~D3D10Effect();

	private:
		D3D10Effect( const EffectConfig& cfg, ResourcePtr eff );

		// manipulation/ access
	public:
		void FillInputSignatureParams( const void*& oInputSignature, UINT32& oByteCodeLength ) const;

		// polymorphism
	private:
		virtual void				SetTechniqueImpl( const AnsiString& name ) OVERRIDE;
		virtual bool				BindImpl( UINT32 passNum ) OVERRIDE;
		virtual EffectVarBindPtr	GetVarBindImpl( const CodeString& name ) OVERRIDE;
		virtual UINT32				GetNumPassesImpl() const;

		// helpers
	private:
		void						CheckTechnique() const;

		// data
	private:
		ResourcePtr		mResource;
		Technique		mCurrentTechnique;
		
	};

}

#endif