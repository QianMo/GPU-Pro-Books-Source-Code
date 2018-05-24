SamplerState pointClampSampler: register(s0);
SamplerState linearClampSampler: register(s1);

Texture2D<float4> depth16Texture: register(t0);
Texture2D<float4> gbufferNormalTexture: register(t1);


#define PI				3.1415f
#define TWO_PI			2.0f * PI
#define GOLDEN_ANGLE	2.4f

#define SAMPLES_COUNT	16


static float2 vogelDiskOffsets16[16] =
{
	float2(0.176777f, 0.0f),
	float2(-0.225718f, 0.206885f),
	float2(0.0343507f, -0.393789f),
	float2(0.284864f, 0.370948f),
	float2(-0.52232f, -0.0918239f),
	float2(0.494281f, -0.315336f),
	float2(-0.164493f, 0.615786f),
	float2(-0.316681f, -0.607012f),
	float2(0.685167f, 0.248588f),
	float2(-0.711557f, 0.295696f),
	float2(0.341422f, -0.73463f),
	float2(0.256072f, 0.808194f),
	float2(-0.766143f, -0.440767f),
	float2(0.896453f, -0.200303f),
	float2(-0.544632f, 0.780785f),
	float2(-0.130341f, -0.975582f)
};


static float2 alchemySpiralOffsets16[16] =
{
	float2(0.19509f, 0.980785f),
	float2(-0.55557f, -0.83147f),
	float2(0.831469f, 0.555571f),
	float2(-0.980785f, -0.195091f),
	float2(0.980785f, -0.19509f),
	float2(-0.83147f, 0.555569f),
	float2(0.555571f, -0.831469f),
	float2(-0.195092f, 0.980785f),
	float2(-0.195089f, -0.980786f),
	float2(0.555569f, 0.83147f),
	float2(-0.831469f, -0.555572f),
	float2(0.980785f, 0.195092f),
	float2(-0.980786f, 0.195088f),
	float2(0.831471f, -0.555568f),
	float2(-0.555572f, 0.831468f),
	float2(0.195093f, -0.980785f)
};


cbuffer ConstantBuffer: register(b0)
{
	float2 pixelSize;
	float2 nearPlaneSize_normalized; // at distance 1 from the eye
	float4x4 viewTransform;
	float aspect;	
	float radius_world;
	float maxRadius_screen;
	float contrast;
}


struct PS_INPUT
{
	float4 position: SV_POSITION;
	float2 texCoord: TEXCOORD0;
};


float3 Position_View(float2 texCoord)
{
	float3 position = float3(texCoord, -1.0f);
	position.xy = position.xy - 0.5f;
	position.y *= -1.0f;
	position.xy *= nearPlaneSize_normalized;
	position *= -depth16Texture.Sample(pointClampSampler, texCoord).x;
	return position;
}


float2 RotatePoint(float2 pt, float angle)
{
	float sine, cosine;
	sincos(angle, sine, cosine);
	
	float2 rotatedPoint;
	rotatedPoint.x = cosine*pt.x + -sine*pt.y;
	rotatedPoint.y = sine*pt.x + cosine*pt.y;
	
	return rotatedPoint;
}


float2 VogelDiskOffset(int sampleIndex, float phi)
{
	float r = sqrt(sampleIndex + 0.5f) / sqrt(SAMPLES_COUNT);
	float theta = sampleIndex*GOLDEN_ANGLE + phi;

	float sine, cosine;
	sincos(theta, sine, cosine);
	
	return float2(r * cosine, r * sine);
}


float2 AlchemySpiralOffset(int sampleIndex, float phi)
{
	float alpha = float(sampleIndex + 0.5f) / SAMPLES_COUNT;
	float theta = 7.0f*TWO_PI*alpha + phi;

	float sine, cosine;
	sincos(theta, sine, cosine);
	
	return float2(cosine, sine);
}


float InterleavedGradientNoise(float2 position_screen)
{
	float3 magic = float3(0.06711056f, 4.0f*0.00583715f, 52.9829189f);
	return frac(magic.z * frac(dot(position_screen, magic.xy)));
}


float AlchemyNoise(int2 position_screen)
{
	return 30.0f*(position_screen.x^position_screen.y) + 10.0f*(position_screen.x*position_screen.y);
}


float4 PSMain(PS_INPUT input): SV_Target0
{
	float2 texCoord00 = input.texCoord + float2(-0.25f, -0.25f)*pixelSize;

	float3 position = Position_View(input.texCoord);
	float3 normal = gbufferNormalTexture.Sample(pointClampSampler, texCoord00).xyz;
	normal = 2.0f*normal - 1.0f;
	normal = mul((float3x3)viewTransform, normal);

	float noise = InterleavedGradientNoise(input.position.xy);
	float alchemyNoise = AlchemyNoise((int2)input.position.xy);

	float2 radius_screen = radius_world / position.z;
	radius_screen = min(radius_screen, maxRadius_screen);
	radius_screen.y *= aspect;
	
	float ao = 0.0f;

	for (int i = 0; i < SAMPLES_COUNT; i++)
	{
		float2 sampleOffset = 0.0f;
		{
		#if (VARIANT == 1)
			sampleOffset = VogelDiskOffset(i, TWO_PI*noise);
		#elif (VARIANT == 2)
			sampleOffset = AlchemySpiralOffset(i, alchemyNoise);
		#elif (VARIANT == 3)
			sampleOffset = vogelDiskOffsets16[i];
			sampleOffset = RotatePoint(sampleOffset, TWO_PI*noise);
		#elif (VARIANT == 4)
			sampleOffset = alchemySpiralOffsets16[i];
			sampleOffset = RotatePoint(sampleOffset, alchemyNoise);
		#endif
		}
		float2 sampleTexCoord = input.texCoord + radius_screen*sampleOffset;

		float3 samplePosition = Position_View(sampleTexCoord);
		float3 v = samplePosition - position;

		ao += max(0.0f, dot(v, normal) + 0.002f*position.z) / (dot(v, v) + 0.001f);
	}

	ao = saturate(ao / SAMPLES_COUNT);
	ao = 1.0f - ao;
	ao = pow(ao, contrast);

	return ao;
}
