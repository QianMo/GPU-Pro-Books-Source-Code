#ifndef BE_SCENE_H
#define BE_SCENE_H

#include "Engine/BindPoints.fx"

#ifdef BE_SCENE_SETUP
	#define BE_SCENE_PREBOUND_S(semantic) semantic
#else
	#define BE_SCENE_PREBOUND_S(semantic) prebound_s(semantic)
#endif

/// Scene texture.
Texture2D SceneTexture : BE_SCENE_PREBOUND_S(bindpoint_s(SceneTarget, t15))
<
	string TargetType = "Permanent";
	string Format = "R16G16B16A16F";
>;

#endif