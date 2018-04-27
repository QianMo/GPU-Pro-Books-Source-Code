#include "Precompiled.h"

#include "Providers/Src/ShaderResourceProvider.h"
#include "Providers/Src/Providers.h"

#include "Common/Src/XIElemAttribute.h"
#include "Common/Src/XIElemArray.h"

#include "SceneRender/Src/EffectParam.h"
#include "SceneRender/Src/EffectParamControllerConfig.h"

#include "EffectParamController_AnimatedTexture.h"

namespace Mod
{
	/*explicit*/
	EffectParamController_AnimatedTexture::EffectParamController_AnimatedTexture( const EffectParamControllerConfig& cfg ) :
	Parent( cfg ),
	mTime( 0 ),
	mFPS( XIFloat( cfg.elem, L"FPS", L"val", 24.f ) )
	{
		struct Convert
		{
			ShaderResourcePtr operator() ( const XMLElemPtr& elem ) const
			{
				return Providers::Single().GetShaderResourceProv()->GetItem( XIAttString( elem, L"val" ) );
			}
		};

		XIElemArray< ShaderResourcePtr, Convert > shaderResources( cfg.elem, L"texture" );
		mResources.swap( shaderResources );
	}

	//------------------------------------------------------------------------

	EffectParamController_AnimatedTexture::~EffectParamController_AnimatedTexture()
	{

	}

	//------------------------------------------------------------------------

	/*virtual*/
	void
	EffectParamController_AnimatedTexture::UpdateImpl( float dt ) /*OVERRIDE*/
	{
		mTime += dt;
		
		float frame = mTime * mFPS;

		mTime  = fmodf( mTime, mResources.size() / mFPS );

		GetConfig().param->SetValue( mResources[ UINT32( frame ) % mResources.size() ] );
	}

	//------------------------------------------------------------------------

	EffectParamControllerPtr CreateEffectParamController_AnimatedTexture( const EffectParamControllerConfig& cfg )
	{
		return EffectParamControllerPtr( new EffectParamController_AnimatedTexture( cfg ) );
	}

}