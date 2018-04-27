#ifndef D3D9DRV_D3D9EFFECTVARBINDCONFIG_H_INCLUDED
#define D3D9DRV_D3D9EFFECTVARBINDCONFIG_H_INCLUDED

#include "Common/Src/VarType.h"

#include "Wrap3D/Src/EffectVarBindConfig.h"
#include "Math/Src/Forw.h"

namespace Mod
{

	struct D3D9EffectVarBindConfig : EffectVarBindConfig
	{
		// types
	public:
		typedef D3DXHANDLE BindType;

		// construction/ destruction
	public:
		D3D9EffectVarBindConfig();

		// data
	public:
		D3DXHANDLE			bind;
		ComPtr<ID3DXEffect>	effect;

		// polymorphism
	private:
		virtual EffectVarBindConfig* Clone() const;
	};

	//------------------------------------------------------------------------

	inline
	D3D9EffectVarBindConfig::D3D9EffectVarBindConfig() :
	bind( NULL )
	{
		type = VarType::UNKNOWN;
	}

	//------------------------------------------------------------------------

	inline
	EffectVarBindConfig*
	D3D9EffectVarBindConfig::Clone() const
	{
		return new D3D9EffectVarBindConfig( *this );
	}

	//------------------------------------------------------------------------	

}

#endif
