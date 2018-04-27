#ifndef EXTRALIB_EFFECTPARAMCONTROLLER_ANIMATEDTEXTURE_H_INCLUDED
#define EXTRALIB_EFFECTPARAMCONTROLLER_ANIMATEDTEXTURE_H_INCLUDED

#include "Forw.h"

#include "Wrap3D/Src/Forw.h"

#include "SceneRender/Src/EffectParamController.h"

namespace Mod
{
	class EffectParamController_AnimatedTexture : public EffectParamController
	{
		// types
	public:
		typedef Types< ShaderResourcePtr > :: Vec	ShaderResourceVec;

		// contruction/ destruction
	public:
		explicit EffectParamController_AnimatedTexture( const EffectParamControllerConfig& cfg );
		~EffectParamController_AnimatedTexture();

		// polymorpism
	private:
		virtual void UpdateImpl( float dt ) OVERRIDE;

		// data
	private:
		float 				mFPS;
		float 				mTime;

		ShaderResourceVec	mResources;
	};
}

#endif