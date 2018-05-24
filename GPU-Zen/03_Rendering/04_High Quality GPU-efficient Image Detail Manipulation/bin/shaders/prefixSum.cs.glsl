#version 430 core

layout(local_size_x = 512) in;

layout(binding = 0, rgba32f)  readonly uniform image2D inImage_0;
layout(binding = 1, rgba32f)  readonly uniform image2D inImage_1;
layout(binding = 2, rgba32f) writeonly uniform image2D outImage_0;
layout(binding = 3, rgba32f) writeonly uniform image2D outImage_1;

shared vec3 shData_0[gl_WorkGroupSize.x * 2];
shared vec3 shData_1[gl_WorkGroupSize.x * 2];

void main(void) {

	uint	id = gl_LocalInvocationID.x;
	ivec2	texCoord = ivec2(id * 2, gl_WorkGroupID.x);
	uint	i0 = id * 2;
	uint	i1 = i0 + 1;

	//	Co-operative Read-in
	//
	shData_0[i0] = imageLoad(inImage_0, texCoord).rgb;
	shData_0[i1] = imageLoad(inImage_0, texCoord + ivec2(1, 0)).rgb;
	shData_1[i0] = imageLoad(inImage_1, texCoord).rgb;
	shData_1[i1] = imageLoad(inImage_1, texCoord + ivec2(1, 0)).rgb;

	barrier();
	memoryBarrierShared();

	//	Prefix-Sum Processing
	//
	const uint loopCount = uint(log2(gl_WorkGroupSize.x)) + 1;

	for (int step = 0; step < loopCount; step++) {

		// offset = 0, 1, 3, 7, 15 ....
		uint offset = (1 << step) - 1;
		uint rIndex = ((id >> step) << (step + 1)) + offset;
		uint wIndex = rIndex + (id & offset) + 1;

		shData_0[wIndex].rgb += shData_0[rIndex].rgb;
		shData_1[wIndex].rgb += shData_1[rIndex].rgb;

		barrier();
		memoryBarrierShared();

	}

	//	Co-operative Write-back
	//
	imageStore( outImage_0, texCoord.yx, vec4(shData_0[i0].rgb, 1.0) );
	imageStore( outImage_0, texCoord.yx + ivec2(0,1), vec4(shData_0[i1].rgb, 1.0) );
	imageStore( outImage_1, texCoord.yx, vec4(shData_1[i0].rgb, 1.0) );
	imageStore( outImage_1, texCoord.yx + ivec2(0,1), vec4(shData_1[i1].rgb, 1.0) );

}
