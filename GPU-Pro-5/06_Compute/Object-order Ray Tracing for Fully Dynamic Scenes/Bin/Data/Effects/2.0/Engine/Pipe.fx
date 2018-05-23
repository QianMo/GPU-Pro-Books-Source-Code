#ifndef BE_PIPE_H
#define BE_PIPE_H

/// Scene texture.
Texture2D LDRTexture : LDRTarget
<
	string TargetType = "Permanent";
	string Format = "R8G8B8A8U_SRGB";
>;

/// Final target.
Texture2D FinalTargetTexture : FinalTarget
<
	string TargetType = "Persistent";
>;

#endif