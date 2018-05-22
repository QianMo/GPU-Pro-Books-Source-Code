#include "Sampling.h"

double Sampling::radicalInverse(int n, int base)
{
	double val = 0.0;
	double invBase = 1.0 / base;
	double invBi = invBase;
	while(n>0)
	{
		// next digit of radical inverse
		int d_i = (n % base);
		val += d_i * invBi;
		n /= base;
		invBi *= invBase;
	}
	return val;
}


double Sampling::foldedRadicalInverse(int n, int base)
{
	double val = 0.0;
	double invBase = 1.0 / base;
	double invBi = invBase;
	int modOffset = 0;
	while(val + base * invBi != val)
	{
		// next digit of folded radical inverse
		int digit = ((n+modOffset)%base);
		val += digit * invBi;
		n /= base;
		invBi *= invBase;
		++modOffset;
	}
	return val;
}


void Sampling::generateHammersleySequence(glm::vec2* sequence, int n)
{
   int start = 1;
	for (int i = start; i < n + start; i++)
	{
		sequence[i-start].x = float(i)/n;
		sequence[i-start].y = static_cast<float>(radicalInverse(i, 2));
	}
}

void Sampling::generateHaltonSequence(glm::vec2* sequence, int n)
{
	for (int i = 1; i <= n; i++)
	{
		sequence[i-1].x = static_cast<float>(radicalInverse(i,2));
		sequence[i-1].y = static_cast<float>(radicalInverse(i,3));
	}
}

void Sampling::generateHaltonSequence(glm::vec2* sequence, int n, int start)
{
	for (int i = start; i < start + n; i++)
	{
		sequence[i-start].x = static_cast<float>(radicalInverse(i,2));
		sequence[i-start].y = static_cast<float>(radicalInverse(i,3));
	}
}

void Sampling::generateHammersleySequence(glm::vec2* sequence, int n, int start)
{
	for (int i = start; i < start + n; i++)
	{
		sequence[i-start].x = float(i)/n;
		sequence[i-start].y = static_cast<float>(radicalInverse(i, 2));
	}
}

void Sampling::generateScrambledHaltonSequenceElement(
   float* sequenceElement, int dimensions, int elementIndex)
{
   for(int d = 0; d < dimensions; d++)
   {
      sequenceElement[d] = 0;//static_cast<float>(foldedRadicalInverse(elementIndex, primes[d]));
   }
}

void Sampling::addScrambledHaltonSequence(
   vector<vector<glm::vec2> >& sampleSets, int d1, int d2, int number)
{
   vector<glm::vec2> samples;
   for(int n = 1; n <= number; n++)
   {
      samples.push_back(glm::vec2(foldedRadicalInverse(n, d1),
         foldedRadicalInverse(n, d2)));
   }
   sampleSets.push_back(samples);
}

void Sampling::setFoldedRadicalInverse(
   float* samples, int dimension1, int dimension2, int number, int startIndex, int& nextStartIndex)
{
   nextStartIndex = number + startIndex;
   for(int i = startIndex; i < nextStartIndex; i++)
   {
      samples[(i - startIndex)*2]   = (float)foldedRadicalInverse(i, dimension1);
      samples[(i - startIndex)*2+1] = (float)foldedRadicalInverse(i, dimension2);
   }
}