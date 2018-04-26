#ifndef _STRUCT_H_
#define _STRUCT_H_

#include <vector>

#include "FRep/structGPU.h"

// stores basic parameters of the FRep model
struct MODEL_PARAMETERS
{  
   
   SUBMODEL_PARAMS   submodelParams;

   // a set of convolution line segments
   std::vector<CONVOLUTION_SEGMENT> segments;
};

// stores a set of sampled model parameters 
struct FREP_DESCRIPTION
{
   POLYGONIZATION_PARAMS   polygonizationParams;

   // indicates whether blending union between convolutions and implicit box ("liquid" object) should be enabled
   bool  isBlendingOn;

   float convolutionIsoValue;

   // sampled parameters of the model for all the frames
   std::vector<MODEL_PARAMETERS> sampledModel;
};

#endif _STRUCT_H_