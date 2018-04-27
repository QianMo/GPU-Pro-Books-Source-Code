#ifndef D3D10DRV_D3D10SHADERRESOURCE_H_INCLUDED
#define D3D10DRV_D3D10SHADERRESOURCE_H_INCLUDED

#include "Wrap3D/Src/ShaderResource.h"

namespace Mod
{
	class D3D10ShaderResource : public ShaderResource
	{
		// types
	public:
		typedef ComPtr<ID3D10Resource>				ResourcePtr;
		typedef ID3D10EffectShaderResourceVariable	BindType1;
		typedef ID3D10ShaderResourceView			BindType2;

		// construction/ destruction
	public:
		D3D10ShaderResource( const ShaderResourceConfig& cfg, ID3D10Device* dev );
		~D3D10ShaderResource();

		// manipulation/ access
	public:
		void			BindTo( BindType1* srv )		const;
		static void		SetBindToZero( BindType1* srv );

		void			BindTo( BindType2* srv )		const;
		static void		SetBindToZero( BindType2* srv );

		// data
	private:
		ComPtr<ID3D10ShaderResourceView>	mSRView;

	};
}

#endif