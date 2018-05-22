
/* * * * * * * * * * * * * Author's note * * * * * * * * * * * *\
*   _       _   _       _   _       _   _       _     _ _ _ _   *
*  |_|     |_| |_|     |_| |_|_   _|_| |_|     |_|  _|_|_|_|_|  *
*  |_|_ _ _|_| |_|     |_| |_|_|_|_|_| |_|     |_| |_|_ _ _     *
*  |_|_|_|_|_| |_|     |_| |_| |_| |_| |_|     |_|   |_|_|_|_   *
*  |_|     |_| |_|_ _ _|_| |_|     |_| |_|_ _ _|_|  _ _ _ _|_|  *
*  |_|     |_|   |_|_|_|   |_|     |_|   |_|_|_|   |_|_|_|_|    *
*                                                               *
*                     http://www.humus.name                     *
*                                                                *
* This file is a part of the work done by Humus. You are free to   *
* use the code in any way you like, modified, unmodified or copied   *
* into your own work. However, I expect you to respect these points:  *
*  - If you use this file and its contents unmodified, or use a major *
*    part of this file, please credit the author and leave this note. *
*  - For use in anything commercial, please request my approval.     *
*  - Share your work and ideas too as much as you can.             *
*                                                                *
\* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "MathTools.h"
#include <stdlib.h>

bool throwDarts(vec2 *samples, const int firstSample, const int nSamples, const float minDistSq, const int maxFailedTries){
	for (int i = firstSample; i < nSamples; i++){
		vec2 sample;
		bool failed;
		int nFailed = 0;
		do {
			do {
				sample = vec2(2 * float(rand()) / RAND_MAX - 1, 2 * float(rand()) / RAND_MAX - 1);
			} while (dot(sample, sample) > 1);

			failed = false;
			for (int k = 0; k < i; k++){
				vec2 d = samples[k] - sample;
				if (dot(d, d) < minDistSq){
					failed = true;

					nFailed++;
					if (nFailed >= maxFailedTries) return false;
					break;
				}
			}

		} while (failed);

		samples[i] = sample;
	}

	return true;
}

bool generatePoissonSamples(vec2 *samples, const int nSamples, const float minDist, const int maxFailedTries, const int nRetries, const bool includeCenter){
	if (includeCenter){
		samples[0] = vec2(0, 0);
	}

	for (int t = 0; t < nRetries; t++){
		if (throwDarts(samples, int(includeCenter), nSamples, minDist * minDist, maxFailedTries)) return true;
	}

	return false;
}
