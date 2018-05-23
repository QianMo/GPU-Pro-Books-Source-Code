#ifndef BE_SHADOW_H
#define BE_SHADOW_H

/// Shadow texture.
Texture2D SceneShadowTexture : SceneShadowTarget
<
	string TargetType = "Permanent";
	string Format = "R32F";
	bool Output = true;
>;

#endif