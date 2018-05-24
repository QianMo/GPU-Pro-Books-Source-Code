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

layout(binding = 0, rgba32f) readonly uniform image2D input;
layout(binding = 1, rgba32f) readonly uniform image2D A_K;
layout(binding = 2, rgba32f) readonly uniform image2D B_K;

layout(binding = 3, rgba32f) writeonly uniform image2D base;
layout(binding = 4, rgba32f) writeonly uniform image2D detail;

uniform int r, lx, ly, ux, uy;

void main(void)
{

	ivec2	coord = ivec2(gl_LocalInvocationID.x, gl_WorkGroupID.x);

	if ( coord.x < lx || coord.x > ux || coord.y < ly || coord.y > uy ) {
		imageStore(   base, coord.xy, vec4(0.0) );
		imageStore( detail, coord.xy, vec4(0.0) );
		return;
	}

	//	Per-pixel filtering
	//
	float	blockSize = r + r + 1;
	float	omega = 1.0 / (blockSize * blockSize);

	vec3	a, b, c, d;
	int r1 = -r - 1;
	
	a = imageLoad(A_K, coord + ivec2( r1, r1 )).rgb;
	b = imageLoad(A_K, coord + ivec2( r1,  r )).rgb;
	c = imageLoad(A_K, coord + ivec2(  r, r1 )).rgb;
	d = imageLoad(A_K, coord + ivec2(  r,  r )).rgb;
	vec3 ai = vec3(omega) * (a - b - c + d);

	a = imageLoad(B_K, coord + ivec2( r1, r1 )).rgb;
	b = imageLoad(B_K, coord + ivec2( r1,  r )).rgb;
	c = imageLoad(B_K, coord + ivec2(  r, r1 )).rgb;
	d = imageLoad(B_K, coord + ivec2(  r,  r )).rgb;
	vec3 bi = vec3(omega) * (a - b - c + d);

	vec3 org = imageLoad(input, coord).rgb;
	vec3 res = ai * org + bi;
	
	//	Decomposition
	//
	imageStore(   base, coord.xy, vec4(res.rgb, 1.0) );
	imageStore( detail, coord.xy, vec4(org.rgb - res.rgb, 1.0) );

}
