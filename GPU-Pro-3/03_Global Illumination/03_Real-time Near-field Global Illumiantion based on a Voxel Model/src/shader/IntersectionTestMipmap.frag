///////////////////////////////////////////////////////////////////////
// Real-time Near-Field Global Illumination based on a Voxel Model
// GPU Pro 3
///////////////////////////////////////////////////////////////////////
// Authors : Sinje Thiedemann, Niklas Henrich, 
//           Thorsten Grosch, Stefan Mueller
// Created:  August 2011
///////////////////////////////////////////////////////////////////////

#version 120
#extension GL_EXT_gpu_shader4 : require 
#define PI 3.14159265359
#define PI2 6.2831853
#define eps 0.001// for TBN basis

// ---------- Uniforms --------- //

uniform sampler2D positionBuffer; // G-buffer texture with world space positions
uniform sampler2D normalBuffer;   // G-buffer texture with world space normals
uniform sampler2D envMap;         // Environment map for directional occlusion
uniform sampler2D randTex;        // small tiled texture holding random values within [0, 1]
uniform usampler2D voxelTexture;  // mipmapped voxel texture
uniform usampler2D bitmaskXORRays; // lookup texture for ray bitmasks

uniform int maxMipMapLevel; // coarsest mipmap level

// transformation matrices 
uniform mat4 viewProjToUnitMatrixVoxelCam;
uniform mat4 inverseViewProjToUnitMatrixVoxelCam;
uniform vec3 voxelizingDirection; // viewing direction of the orthographic voxelization camera

uniform float voxelDiagonal; // length of a voxel diagonal in world coordinates

// User defines offset scaling factors.
// Offsets are scene-dependent, 
// prevent self-shadowing.
uniform float voxelOffsetCosThetaScale;
uniform float voxelOffsetNormalScale;
float cosThetaScale = voxelOffsetCosThetaScale > 0.0 ? voxelOffsetCosThetaScale : 1.2;

uniform vec2 samplingSequence[128];// 2d hammersley sequence
uniform int currentRay; // index of the current ray to be traced

uniform float radius; // maximum length of rays
uniform int steps; // maximum number of traversal iterations
uniform float spread; // a value of 1.0 = full hemisphere; smaller: 

uniform float useRandRay; // randomly rotate rays 
uniform int randTexSize; // width and height of a small square tiled texture 
                         // that is used to randomly rotate the rays per pixel

uniform float occlusionStrength;
uniform float envMapBrightness;
uniform float lightRotationAngle; // rotation of environment map
uniform float invRays; // 1 / # rays per pixel

// ---------- Global variables --------- //
vec3 origin;	// ray origin 
vec3 dir;		// ray direction (end-origin)

// ---------- The result --------- //
varying out vec3 hitPos;
varying out vec3 directionalOcclusion;


// ---------- Helper functions --------- //

//see http://blog.piloni.net/?p=114
/* Bounding box intersection routine [Slabs]
*  uses globally defined ray origin and direction
*  @param vec3 box_min  Box’s minimum extent coordinates
*  @param vec3 box_max  Box’s maximum extent coordinates
*  @return true if an intersection was found, false otherwise
*   minmax = most far intersection with the box
*   maxmin = nearest intersection with the box
*/

bool IntersectBox(in vec3 box_min, in vec3 box_max, /*out float maxmin, */out float minmax)
{
	vec3 tmin = (box_min - origin) / dir; 
	vec3 tmax = (box_max - origin) / dir; 

	vec3 real_min = min(tmin,tmax); 
	vec3 real_max = max(tmin,tmax);

	minmax = min(1.0, min( min(real_max.x, real_max.y), real_max.z)); // the minimal maximum is tFar
	float maxmin = max(0.0, max( max(real_min.x, real_min.y), real_min.z)); // the maximal minimum is tNear 

	return (minmax >= maxmin); // hit the box? 
}

// a minimal version of the method above
// (we assume: ray always hits the box)
float IntersectBoxOnlyTFar(in vec3 box_min, in vec3 box_max)
{
	vec3 tmin = (box_min - origin) / dir; 
	vec3 tmax = (box_max - origin) / dir; 

	vec3 real_max = max(tmin,tmax);

   // the minimal maximum is tFar
   // clamp to 1.0
   return min(1.0, min( min(real_max.x, real_max.y), real_max.z)); 
}


vec3 worldToUnit(in vec3 p)
{
   return (viewProjToUnitMatrixVoxelCam * vec4(p, 1.0)).xyz;
}

bool intersectBits(in uvec4 bitRay, in ivec2 texel, in int level, out uvec4 intersectionBitmask)
{
   // Fetch bitmask from hierarchy and compute intersection via bitwise AND
   intersectionBitmask = (bitRay & texelFetch2D(voxelTexture, texel, level));
	return (intersectionBitmask != uvec4(0));
}


bool IntersectHierarchy(in int level, in vec3 posTNear, inout float tFar, out uvec4 intersectionBitmask)
{
	// Calculate pixel coordinates ([0,width]x[0,height]) 
   // of the current position along the ray
   float res = float(1 << (maxMipMapLevel - level));
   ivec2 pixelCoord = ivec2(posTNear.xy * res);

   // Voxel width and height in the unit cube
   vec2 voxelWH = vec2(1.0) / res;

	// Compute voxel stack (AABB) in the unit cube
   // belonging to this pixel position
   // (Slabs for AABB/Ray Intersection)
	vec2 box_min = pixelCoord * voxelWH; // (left, bottom)

	// Compute intersection with the bounding box
	// It is always assumed that an intersecion occurs
	// It is assumed that the position of posTNear always remains the same
	tFar = IntersectBoxOnlyTFar(
      vec3(box_min, 0.0), 
      vec3(box_min + voxelWH, 1.0));

	// Now test if some of the bits intersect
	float zFar = tFar*dir.z + origin.z ;

	// Fetch bit-mask for ray and intersect with current pixel
	return intersectBits(
      texture2D(bitmaskXORRays, // stores all possible bitmask
      vec2(min(posTNear.z, zFar), max(posTNear.z, zFar))),
      pixelCoord, level, intersectionBitmask);	
}


void main()
{
	vec3 P_world = texture2D(positionBuffer, gl_TexCoord[0].xy).xyz; // this pixel's world space coord.
   
   hitPos = vec3(0.0, 0.0, 100.0); // no hit
   directionalOcclusion = vec3(0);

	// geometry rendered to buffer?
	if(P_world.z < 100.0)
   {
		// TBN Basis
      vec3 N = texture2D(normalBuffer, gl_TexCoord[0].xy).xyz;
      //arbitrary vector H not coinciding with the N-Vector
      vec3 H = vec3(0.0, 0.0, 1.0);
      if(abs(N.x) < eps && abs(N.y) < eps)
         H = vec3(0.0, 1.0, 0.0);
      vec3 BiTan = normalize(cross(N, H)); // Bitangent
      vec3 Tan = cross(BiTan, N); // Tangent


      vec2 rand = texelFetch2D(randTex, ivec2(mod(ivec2(gl_FragCoord.st), ivec2(randTexSize))), 0).rg;

      // construct ray
      float u1 = fract(samplingSequence[currentRay].x + useRandRay * rand.x);
      float u2 = fract(samplingSequence[currentRay].y + useRandRay * rand.y);

      // cosine weighted sampling (pdf = cos / PI)
      float r = sqrt(u1*spread);  
      float phi = 2 * PI * u2;  

      float x = cos(phi) * r;
      float y = sin(phi) * r;
      float z = sqrt(max(0.0, 1.0 - x*x - y*y)); // sqrt(1.0 - u1 * spread)

      vec3 ray = normalize(x*Tan + y*BiTan + z*N);

      // To prevent self-shadowing, the starting point of
      // the ray is advanced by an offset.
      // The idea is to move the starting point at least
      // the size of the voxel diagonal. Furthermore we
      // want to consider the angle between the ray and the
      // normal of the starting point. The greater the angle,
      // the higher the chances that the ray hits
      // the voxelized surface it starts from. 
      // In this case, the offset should be greater 
      // than if the ray lies in the direction of
      // the normal.
      float startOffset = cosThetaScale * voxelDiagonal / dot(ray, N);  
      if(startOffset <= radius) // else ignore this ray
      {
         // Start and end point of the ray segment in world coordinates:
         origin   = P_world + voxelOffsetNormalScale * voxelDiagonal * N + startOffset * ray;
         vec3 end = P_world + voxelOffsetNormalScale * voxelDiagonal * N + radius * ray;
   
         // Only start the visibility test if the ray segment to be tested
         // has a sufficient length compared to the voxel size.
         if(distance(origin, end) > voxelDiagonal*1.5)
         {
            origin = worldToUnit(origin);
            dir    = worldToUnit(end) - origin;

            // Adjust direction a bit to prevent division by zero
            dir.x = (abs(dir.x) < 0.0000001) ? 0.0000001 : dir.x;
            dir.y = (abs(dir.y) < 0.0000001) ? 0.0000001 : dir.y;
            dir.z = (abs(dir.z) < 0.0000001) ? 0.0000001 : dir.z;

            // Compute offset for hierarchy traversal steps.
            // The ray is parameterized with tNear = 0 = origin of the ray
            // and tFar = 1 = endpoint of the ray, independent of the actual length
            // of the ray in world-coordinates. The offset is used to advance
            // tNear and ensures that a neighboring voxel stack is tested
            // in the next traversal step.
            // The offset should depend on the actual length of the ray. 
            // An offset independent of the actual length of the ray would
            // result in a larger offset in world-coordinates for a short ray
            // and in a smaller offset for long rays. Dividing by the actual ray
            // length keeps the offset constant in world-coordinates.
            float offset = (0.25 / (1 << maxMipMapLevel)) / length(dir); 

            float tNear = 0.0;
            float tFar;
            
            // Compute the exit position of the ray with the scene's bounding box
            if(!IntersectBox(vec3(0.0), vec3(1.0), tFar))
               // IntersectBox modifies tFar, so set a proper value
               tFar = 1.0; // test whole ray
            // else: test only ray part

            // Set current position along the ray to the ray's origin
            vec3 posTNear = origin;

            bool intersectionFound = false;			
            uvec4 intersectionBitmask = uvec4(0);

            /// Choose the mipmap level for the initial iteration.
            /// Do not start at a too fine level, but at most at the
            /// immediate children of the root node, because the intersection
            /// with the root node was already computed
            int level = min(3, maxMipMapLevel-1); // maxMipMapLevel / 2;  

            for(int i = 0; (i < steps) && (tNear <= tFar) && (!intersectionFound); i++)			
            {		
               float newTFar = 1.0f;						

               if(IntersectHierarchy(level, posTNear, newTFar, intersectionBitmask))
               {	
                  // If we are at mipmap level 0 and an intersection occured,
                  // we have found an intersection of the ray with the voxels
                  intersectionFound = (level == 0);

                  // Otherwise we have to move down one level and
                  // start testing from there
                  level --;				
               }
               else
               {
                  // If no intersection occurs, we have to advance the
                  // position on the ray to test the next element of the hierachy.
                  // Furthermore, add a small offset computed beforehand to
                  // handle floating point inaccuracy.
                  tNear = newTFar + offset;
                  posTNear = origin + tNear * dir;

                  // Move one level up
                  level ++;
               }
            }	

            //
            // Have we found an intersection?
            // If so, find the real hit position
            //

            if(intersectionFound)
            {

               // Compute the position of the highest or lowest set bit in the resulting bitmask.
               // Compute highest bit if ray is not reversed to the voxelization direction,
               // lowest if reversed.
               bool reversed = dot(ray, voxelizingDirection) < 0.0;
               int bitPosition = 0; 
               int x = 0;
               if(!reversed)
               {
                  // get the position of the highest bit set
                  int v;
                  for(v = 0; x == 0 && v < 4; v++) // r g b a
                  {
                     x = int(intersectionBitmask[v]);
                     if(x != 0)
                     {
                        int pos32 = int(log2(float(x)));
                        bitPosition = (3-v)*32 + pos32;
                     }
                  }

               }
               else
               {
                  // get the position of the lowest bit set
                  int v;
                  for(v = 3; x == 0 && v >= 0; v--) // r g b a
                  {
                     x = int(intersectionBitmask[v]);
                     if(x != 0)
                     {
                        int pos32 = int(log2(float(x & ~(x-1)))+0.1);
                        bitPosition = (3-v)*32 + pos32;
                     }
                  }

               }
               posTNear.z = float(127 - bitPosition) * 0.0078125  + 0.00390625; 
               // 0.0078125 = 1/128 (length of one voxel in z-direction in unit-coordinates)
               // 0.00390625 = 0.5/128 (half voxel)
               hitPos = (inverseViewProjToUnitMatrixVoxelCam * vec4(posTNear, 1.0)).xyz;
            } // intersectionFound


            // Read sender radiance from environment map
            // Compute directional occlusion

            float envTheta = acos(ray.y);              
            float envPhi = atan(ray.z, ray.x);
            envPhi += lightRotationAngle;
            if (envPhi < 0.0) envPhi += PI2;
            if (envPhi > PI2) envPhi -= PI2;

            vec3 senderRadiance = texture2D(envMap, vec2( envPhi / (PI2), 1.0 - envTheta  / PI ) ).rgb;

            float receiverGeometricTerm = max(0.0, dot(ray, N));
            float visibility = 1.0 - (intersectionFound ? occlusionStrength : 0.0); 
            directionalOcclusion = envMapBrightness * invRays * visibility * receiverGeometricTerm * senderRadiance;

         } // mip map test


      }

	}

}

