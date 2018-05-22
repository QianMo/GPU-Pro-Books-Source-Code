// GUI adjustables
float lightx <
	string SasUiControl = "Slider";
	string SasUiLabel = "light x";
	float SasUiMin = -450;
	float SasUiMax = 450;
	int SasUiSteps = 200;
>  = -450;
float lighty <
	string SasUiControl = "Slider";
	string SasUiLabel = "light y";
	float SasUiMin = -450;
	float SasUiMax = 450;
	int SasUiSteps = 200;
>  = 450;

float lightz <
	string SasUiControl = "Slider";
	string SasUiLabel = "light z";
	float SasUiMin = -450;
	float SasUiMax = 450;
	int SasUiSteps = 200;
>  = 0;

float surfaceReflectionWeight <
	string SasUiControl = "Slider";
	string SasUiLabel = "Surface reflection";
	float SasUiMin = 0;
	float SasUiMax = 1;
	int SasUiSteps = 200;
>  = 0;

/// viewport size
uint2 frameDimensions;
/// deferred buffer as shader input
Texture2D opaqueTexture;

/// Transparent materials
cbuffer transparentMaterialBuffer
{
	struct TransparentMaterial
	{
		float4 gtau;
		float4 lighting;
	} transparentMaterials[7] =
	{
		{ float4(0.0, 0.0, 0.0, 0.0), float4(-1, -1, -1, -1)},
		{ float4(0.1, 0.0, 0.0, 0.35), float4(0, 0, 0, 0)},
		{ float4(0.0, 0.05, 0.05, 0.35), float4(0, 0, 0, 0)},
		{ float4(0.007, 0.0, 0.007, 0.03), float4(0, 0, 0, 0)},
		{ float4(0.004, 0.0, 0.004, 0.02), float4(0, 0, 0, 0)},
		{ float4(0.02, 0.02, 0.0, 0.02), float4(0, 0, 0, 0)},
		{ float4(0.0, 0.002, 0.002, 0.02), float4(0, 0, 0, 0)},
	};
}