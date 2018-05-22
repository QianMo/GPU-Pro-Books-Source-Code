#ifndef SAMPLING_H
#define SAMPLING_H

#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <vector>

#include "glm/glm.hpp"

using namespace std;

class Sampling
{
public:
   // fills the 2D-Sequence array with n pairs of the Hammersley Sequence
   static void generateHammersleySequence(glm::vec2* sequence, int n);

   // fills the 2D-Sequence array with n pairs of the Halton Sequence
   static void generateHaltonSequence(glm::vec2* sequence, int n);
   // fills the 2D-Sequence array with n pairs of the Halton Sequence
   // start computing the sequence at i = start
   static void generateHaltonSequence(glm::vec2* sequence, int n, int start);
   static void generateHammersleySequence(glm::vec2* sequence, int n, int start);

   // see Phyiscally Based Rendering, page 319ff
   static double radicalInverse(int n, int base);

   // the array result is an array of with "dimensions" elements
   static void generateScrambledHaltonSequenceElement(
      float* sequenceElement, 
      int dimensions, int elementIndex);

   static void addScrambledHaltonSequence(
      vector<vector<glm::vec2> >& sampleSets, int d1, int d2, int number);

   static void setFoldedRadicalInverse(float* samples, 
      int dimension1, int dimension2,
      int number, int startIndex, int& nextStartIndex);

private:
   Sampling(){};

   // taken from Phyiscally Based Rendering, page 319ff
   static double foldedRadicalInverse(int n, int base);
};

#endif
