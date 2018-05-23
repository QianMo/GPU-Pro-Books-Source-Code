#ifndef BE_LPR_SCENE_H
#define BE_LPR_SCENE_H

#include "Pipelines/Scene.fx"
#include "Engine/BindPoints.fx"

/// Scene depth texture.
Texture2D SceneGeometryTexture : BE_SCENE_PREBOUND_S(bindpoint_s(SceneGeometryTarget, t14))
<
	string TargetType = "Permanent";
	string Format = "R16G16B16A16F";
>;

/// Scene texture.
Texture2D SceneDiffuseTexture : BE_SCENE_PREBOUND_S(bindpoint_s(SceneDiffuseTarget, t13))
<
	string TargetType = "Permanent";
	string Format = "R8G8B8A8U_SRGB";
>;

/// Scene texture.
Texture2D<uint> SceneSpecularTexture : BE_SCENE_PREBOUND_S(bindpoint_s(SceneSpecularTarget, t12))
<
	string TargetType = "Permanent";
	string Format = "R32U";
>;

/// Depth buffer.
Texture2D SceneDepthBuffer : BE_SCENE_PREBOUND_S(SceneDepthBuffer)
<
	string TargetType = "Permanent";
	string Format = "D24S8";
>;

#endif