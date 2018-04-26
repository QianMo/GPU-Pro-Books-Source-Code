/*
**********************************************************************
 * Demo program for
 * Rule-based Geometry Synthesis in Real-time
 * ShaderX 8 article.
 *
 * @author: Milan Magdics
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted for any non-commercial programs.
 * 
 * Use it for your own risk. The author(s) do(es) not take
 * responsibility or liability for the damages or harms caused by
 * this software.
**********************************************************************
*/

// some shared global stuff for instancing

#ifndef INSTANCING_HFX
#define INSTANCING_HFX

//******************
// Global variables
//******************

shared cbuffer transformations
{
	// matrices
	float4x4 modelMatrix;
	float4x4 modelMatrixInverse;
	float4x4 modelViewProjMatrix;
	float4x4 viewProjMatrix;
}

shared Texture2D txFloor;
shared Texture2D txWall;

shared SamplerState samLinear
{
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Wrap;
    AddressV = Wrap;
};

#endif