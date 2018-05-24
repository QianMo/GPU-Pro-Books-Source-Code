#pragma once


#include "types.h"
#include "vector.h"


namespace NMath
{
	vector<Vector2> VogelDiskSamples(uint n);
	vector<Vector2> AlchemySpiralSamples(uint n, uint spiralsCount);

	//

	inline vector<Vector2> VogelDiskSamples(uint n)
	{
		vector<Vector2> samples;
		samples.resize(n);

		for (uint i = 0; i < n; i++)
		{
			float r = sqrt(i + 0.5f) / sqrt((float)n);
			float theta = i * GoldenAngle;

			samples[i].x = r * Cos(theta);
			samples[i].y = r * Sin(theta);
		}

		return samples;
	}

	inline vector<Vector2> AlchemySpiralSamples(uint n, uint spiralsCount)
	{
		vector<Vector2> samples;
		samples.resize(n);

		for (uint i = 0; i < n; i++)
		{
			float alpha = float(i + 0.5f) / (float)n;
			float theta = spiralsCount * TwoPi * alpha;

			samples[i].x = Cos(theta);
			samples[i].y = Sin(theta);
		}

		return samples;
	}
}
