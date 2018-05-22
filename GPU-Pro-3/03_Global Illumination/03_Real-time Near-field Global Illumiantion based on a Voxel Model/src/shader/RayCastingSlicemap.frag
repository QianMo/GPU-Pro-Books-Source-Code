#version 120
#extension GL_EXT_gpu_shader4 : require

uniform sampler2D texPositionRayStart;
uniform sampler2D texPositionRayEnd;

uniform usampler2D voxelTexture; // level 0: width x height, ..., level n: 1x1
uniform usampler1D bitmask; 


uniform mat4 viewMatrixVoxelCam;
uniform mat4 projMatrixVoxelCam;
uniform mat4 inverseViewMatrixUserCam;

uniform bool writeEyePos;

varying out vec4 hitPos;

bool intersectBits(in uvec4 bitRay, in vec2 texel)
{
	uvec4 result = bitRay & texture2DLod(voxelTexture, texel, 0);		
	return (result != uvec4(0));
}


vec3 voxelEyeToWindow(in vec3 p)
{
	vec4 clipCoord = projMatrixVoxelCam * vec4(p, 1.0);
	return clipCoord.xyz * 0.5 + vec3(0.5); 
}



void main()
{
	hitPos = vec4(1.0);
   hitPos.z = 100.0;

	vec4 rayStart = texture2D(texPositionRayStart, gl_TexCoord[0].st);

	if(rayStart.w != 0.0) // w == 1.0: positon written to texture
   {
		vec4 rayEnd = texture2D(texPositionRayEnd, gl_TexCoord[0].st);

		rayStart = viewMatrixVoxelCam * inverseViewMatrixUserCam * rayStart;
		rayEnd   = viewMatrixVoxelCam * inverseViewMatrixUserCam * rayEnd;

		vec3 rayDir = rayEnd.xyz - rayStart.xyz;

      int steps = textureSize2D(voxelTexture, 0).x * 2; 
		float inc = 1.0 / float(steps);
      float t = 0.0;
		for(t = 0.0f; t < 1.0f; t += inc)
      {
			vec3 posOnRay = rayStart.xyz + (t * rayDir);
			vec3 viewPortPos = voxelEyeToWindow(posOnRay);

			if(intersectBits(texture1D(bitmask, viewPortPos.z), viewPortPos.st))
         {
            hitPos = vec4(writeEyePos ? posOnRay : viewPortPos, 1);
				break;
			}
		}
	}


}