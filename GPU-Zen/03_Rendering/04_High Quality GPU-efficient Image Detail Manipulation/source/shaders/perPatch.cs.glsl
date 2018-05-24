#version 430 core

//
//	Copyright (c) 2016, Kin-Ming Wong and Tien-Tsin Wong
//  All rights reserved.
//
//	Redistribution and use in source and binary forms, with or without modification,
//	are permitted provided that the following conditions are met:
//
//	Redistributions of source code must retain the above copyright notice,
//	this list of conditions and the following disclaimer.
//
//	Redistributions in binary form must reproduce the above copyright notice,
//	this list of conditions and the following disclaimer in the documentation
//	and/or other materials provided with the distribution.
//
//	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
//	ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//	WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
//	IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
//	INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
//	BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
//	DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//	LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
//	OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
//	OF THE POSSIBILITY OF SUCH DAMAGE.
//
//	Please cite our original article
//	'High Quality GPU-efficient Image Detail Manipulation'
//	in GPU Zen if you use any part of the following code.
//

layout(local_size_x = 1024) in;

layout(binding = 0, rgba32f)  readonly uniform image2D sat_I2;
layout(binding = 1, rgba32f)  readonly uniform image2D sat_I;
layout(binding = 2, rgba32f) writeonly uniform image2D Ak;
layout(binding = 3, rgba32f) writeonly uniform image2D Bk;

uniform int r;
uniform float epsilon;

void compVar(in ivec2 coord, in int lx, in int ly, in int ux, in int uy, in float omega, out vec3 variance)
{

	ivec2 P0 = coord + ivec2(lx, ly);
	ivec2 P1 = coord + ivec2(lx, uy);
	ivec2 P2 = coord + ivec2(ux, ly);
	ivec2 P3 = coord + ivec2(ux, uy);

	vec3	a, b, c, d;

	a = imageLoad(sat_I2, P0).rgb;
	b = imageLoad(sat_I2, P1).rgb;
	c = imageLoad(sat_I2, P2).rgb;
	d = imageLoad(sat_I2, P3).rgb;
	vec3	u2 = vec3(omega) * (a - b - c + d);

	a = imageLoad(sat_I, P0).rgb;
	b = imageLoad(sat_I, P1).rgb;
	c = imageLoad(sat_I, P2).rgb;
	d = imageLoad(sat_I, P3).rgb;
	vec3	u = vec3(omega) * (a - b - c + d);

	variance = u2 - (u * u);

};


void main(void)
{

	ivec2	coord = ivec2(gl_LocalInvocationID.x, gl_WorkGroupID.x);

	//	Whole-window mean:uG and variance:vG
	//
	float	blockSize = r + r + 1;
	float	omega = 1.0 / (blockSize * blockSize);

	int		lx = -r - 1;
	int		ux = r;
	int		ly = -r - 1;
	int		uy = r;

	vec3	a, b, c, d;

	a = imageLoad(sat_I, coord + ivec2( lx, ly )).rgb;
	b = imageLoad(sat_I, coord + ivec2( lx, uy )).rgb;
	c = imageLoad(sat_I, coord + ivec2( ux, ly )).rgb;
	d = imageLoad(sat_I, coord + ivec2( ux, uy )).rgb;

	vec3	uG = vec3(omega) * (a - b - c + d);
	vec3	vG;	
	compVar(coord, lx, ly, ux, uy, omega, vG);

	
	//	Sub-window variances
	//
	vec3	vA, vB, vC, vD;
	blockSize = r + 1;
	omega = 1.0 / (blockSize * blockSize);

	//	sub-window A
	ux = 0;
	uy = 0;
	compVar(coord, lx, ly, ux, uy, omega, vA);

	//	sub-window B
	ly = -1;
	uy = r;
	compVar(coord, lx, ly, ux, uy, omega, vB);

	//	sub-window C 
	//
	lx = -1;
	ux = r;
	ly = -r - 1;
	uy = 0;
	compVar(coord, lx, ly, ux, uy, omega, vC);

	//	sub-window D
	ly = -1;
	uy = r;
	compVar(coord, lx, ly, ux, uy, omega, vD);

	//	Compute per-patch preservation
	//
	vec3	eps, vP, vMin, vMax;

	vMin = min( min(vA, vB), min(vC, vD) );
	vMax = max( vG, max(max(vA, vB), max(vC, vD)) );
	eps	 = vec3(epsilon);

	vec3	ak;
	ak = vMax / (eps + vMin);
	ak = min(ak, vec3(1.0));

	//	Maximum ak among 3 channels
	//
	float akMax;
	akMax = max(ak.r, max(ak.g, ak.b));
	ak = vec3(akMax);

	//	Compute bk
	//
	vec3	bk	= uG * (vec3(1.0) - ak);

	imageStore( Ak, coord.xy, vec4(ak.rgb, 1.0) );
	imageStore( Bk, coord.xy, vec4(bk.rgb, 1.0) );

}
