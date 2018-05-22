
struct Wave
{
	float		wavelength;
	float		amplitude;
	float3		direction;
};

#define NWAVES 4
Wave wave[NWAVES] = { {130.0f, 8.0f, float3(-0.7, 0, 0.7) },
						{70.5f, 7.0f, float3(1, 0, -0.0) },
						{50.0f, 5.0f, float3(-0.7, 0, -0.7) },
						{20.0f, 4.0f, float3(0.5, 0, 0.8) }
};

float3 evaluateWaveWithDerivative(Wave w, float phase, out float3 derivative)
{
	float2 offset;
	sincos(phase, offset.x, offset.y);
	float amplitude = w.amplitude;
	offset *= -amplitude;

	derivative = (w.direction + float3(0, 1, 0)) * offset.yxy;

	float3 displacement = (w.direction - float3(0, 1, 0)) * offset.xyx;

	return displacement;
}

float time=0;

VsosTrafo vsFluid(IaosTrafo input)
{
	VsosTrafo output = (VsosTrafo)0;
		
	float3 posShift = 0;
	float3 du = float3(1, 0, 0);
	float3 dv = float3(0, 0, 1);
	float3 da;

	for (int i=0;i<NWAVES;i++)
	{
		float kdotp = dot(input.pos.xz, wave[i].direction.xz);
		float phase = 6.28 / wave[i].wavelength * (
			kdotp 
			+ (time + materialId * 7.85) * sqrt(10.5915 * wave[i].wavelength));
		posShift += evaluateWaveWithDerivative(wave[i], phase, da);
		da *= 6.28 / wave[i].wavelength;
		du += da * wave[i].direction.x;
		dv += da * wave[i].direction.z;
	}
	
	input.pos.xyz += posShift;

	output.normal   = normalize(cross(dv,du));
	output.worldPos = mul(input.pos, modelMatrix);
	output.pos = mul(input.pos, modelViewProjMatrix);
	output.tex = input.pos.xz;

	return output;
}
