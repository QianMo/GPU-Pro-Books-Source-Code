#pragma once

#include <string>

enum EffectMode
{
	EFFECT_SHADING,
	EFFECT_SHADOWMAP,
	EFFECT_DEEPSHADOWMAP,
	EFFECT_EXPDEEPSHADOWMAP,
};

#define NUM_EFFECTS (EFFECT_EXPDEEPSHADOWMAP + 1)

extern std::wstring EffectPaths[NUM_EFFECTS];