/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_LIGHT_EFFECT_DATA
#define BE_SCENE_LIGHT_EFFECT_DATA

#include "beScene.h"
#include <beMath/beMatrixDef.h>
#include <beGraphics/beTexture.h>

namespace beScene
{

/// Typical light data structure.
struct LightEffectData
{
	uint4 TypeID;
	beGraphics::TextureViewHandle Lights;
	beGraphics::TextureViewHandle Shadows;
	beGraphics::TextureViewHandle *ShadowMaps;
	uint4 ShadowMapCount;

	/// Default constructor.
	LightEffectData()
		: TypeID( static_cast<uint4>(-1) ),
		ShadowMapCount(0) { }
};

} // namespace

#endif