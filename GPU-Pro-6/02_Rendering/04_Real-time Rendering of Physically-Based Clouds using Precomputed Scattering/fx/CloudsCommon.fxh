//--------------------------------------------------------------------------------------
// Copyright 2013 Intel Corporation
// All Rights Reserved
//
// Permission is granted to use, copy, distribute and prepare derivative works of this
// software for any purpose and without fee, provided, that the above copyright notice
// and this statement appear in all copies.  Intel makes no representations about the
// suitability of this software for any purpose.  THIS SOFTWARE IS PROVIDED "AS IS."
// INTEL SPECIFICALLY DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, AND ALL LIABILITY,
// INCLUDING CONSEQUENTIAL AND OTHER INDIRECT DAMAGES, FOR THE USE OF THIS SOFTWARE,
// INCLUDING LIABILITY FOR INFRINGEMENT OF ANY PROPRIETARY RIGHTS, AND INCLUDING THE
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  Intel does not
// assume any responsibility for any errors which may appear in this software nor any
// responsibility to update it.
//--------------------------------------------------------------------------------------

#ifndef OPTICAL_DEPTH_LUT_DIM
#   define OPTICAL_DEPTH_LUT_DIM float4(64,32,64,32)
#endif

#ifndef NUM_PARTICLE_LAYERS
#   define NUM_PARTICLE_LAYERS 1
#endif

#ifndef SRF_SCATTERING_IN_PARTICLE_LUT_DIM
#   define SRF_SCATTERING_IN_PARTICLE_LUT_DIM float3(32,64,16)
#endif

#ifndef VOL_SCATTERING_IN_PARTICLE_LUT_DIM
#   define VOL_SCATTERING_IN_PARTICLE_LUT_DIM float4(32,64,32,8)
#endif

#ifndef THREAD_GROUP_SIZE
#   define THREAD_GROUP_SIZE 64
#endif

// Computes direction from the zenith and azimuth angles in XZY (Y Up) coordinate system
float3 ZenithAzimuthAngleToDirectionXZY(in float fZenithAngle, in float fAzimuthAngle)
{
    //       Y   Zenith
    //       |  /
    //       | / /'
    //       |  / '
    //       | /  '
    //       |/___'________X
    //      / \  -Azimuth
    //     /   \  '
    //    /     \ '
    //   Z       \'

    float fZenithSin, fZenithCos, fAzimuthSin, fAzimuthCos;
    sincos(fZenithAngle,  fZenithSin,  fZenithCos);
    sincos(fAzimuthAngle, fAzimuthSin, fAzimuthCos);

    float3 f3Direction;
    f3Direction.y = fZenithCos;
    f3Direction.x = fZenithSin * fAzimuthCos;
    f3Direction.z = fZenithSin * fAzimuthSin;
    
    return f3Direction;
}

// Computes the zenith and azimuth angles in XZY (Y Up) coordinate system from direction
void DirectionToZenithAzimuthAngleXZY(in float3 f3Direction, out float fZenithAngle, out float fAzimuthAngle)
{
    float fZenithCos = f3Direction.y;
    fZenithAngle = acos(fZenithCos);
    //float fZenithSin = sqrt( max(1 - fZenithCos*fZenithCos, 1e-10) );
    float fAzimuthCos = f3Direction.x;// / fZenithSin;
    float fAzimuthSin = f3Direction.z;// / fZenithSin;
    fAzimuthAngle = atan2(fAzimuthSin, fAzimuthCos);
}

// Constructs local XYZ (Z Up) frame from Up and Inward vectors
void ConstructLocalFrameXYZ(in float3 f3Up, in float3 f3Inward, out float3 f3X, out float3 f3Y, out float3 f3Z)
{
    //      Z (Up)
    //      |    Y  (Inward)
    //      |   /
    //      |  /
    //      | /  
    //      |/
    //       -----------> X
    //
    f3Z = normalize(f3Up);
    f3X = normalize(cross(f3Inward, f3Z));
    f3Y = normalize(cross(f3Z, f3X));
}

// Computes direction in local XYZ (Z Up) frame from zenith and azimuth angles
float3 GetDirectionInLocalFrameXYZ(in float3 f3LocalX, 
                                in float3 f3LocalY, 
                                in float3 f3LocalZ,
                                in float fLocalZenithAngle,
                                in float fLocalAzimuthAngle)
{
    // Compute sin and cos of the angle between ray direction and local zenith
    float fDirLocalSinZenithAngle, fDirLocalCosZenithAngle;
    sincos(fLocalZenithAngle, fDirLocalSinZenithAngle, fDirLocalCosZenithAngle);
    // Compute sin and cos of the local azimuth angle
    
    float fDirLocalAzimuthCos, fDirLocalAzimuthSin;
    sincos(fLocalAzimuthAngle, fDirLocalAzimuthSin, fDirLocalAzimuthCos);
    // Reconstruct view ray
    return f3LocalZ * fDirLocalCosZenithAngle + 
           fDirLocalSinZenithAngle * (fDirLocalAzimuthCos * f3LocalX + fDirLocalAzimuthSin * f3LocalY );
}

// Computes zenith and azimuth angles in local XYZ (Z Up) frame from the direction
void ComputeLocalFrameAnglesXYZ(in float3 f3LocalX, 
                             in float3 f3LocalY, 
                             in float3 f3LocalZ,
                             in float3 f3RayDir,
                             out float fLocalZenithAngle,
                             out float fLocalAzimuthAngle)
{
    fLocalZenithAngle = acos(saturate( dot(f3LocalZ, f3RayDir) ));

    // Compute azimuth angle in the local frame
    float fViewDirLocalAzimuthCos = dot(f3RayDir, f3LocalX);
    float fViewDirLocalAzimuthSin = dot(f3RayDir, f3LocalY);
    fLocalAzimuthAngle = atan2(fViewDirLocalAzimuthSin, fViewDirLocalAzimuthCos);
}

void WorldParamsToOpticalDepthLUTCoords(in float3 f3NormalizedStartPos, in float3 f3RayDir, out float4 f4LUTCoords)
{
    DirectionToZenithAzimuthAngleXZY(f3NormalizedStartPos, f4LUTCoords.x, f4LUTCoords.y);

    float3 f3LocalX, f3LocalY, f3LocalZ;
    // Construct local tangent frame for the start point on the sphere (z up)
    // For convinience make the Z axis look into the sphere
    ConstructLocalFrameXYZ( -f3NormalizedStartPos, float3(0,1,0), f3LocalX, f3LocalY, f3LocalZ);

    // z coordinate is the angle between the ray direction and the local frame zenith direction
    // Note that since we are interested in rays going inside the sphere only, the allowable
    // range is [0, PI/2]

    float fRayDirLocalZenith, fRayDirLocalAzimuth;
    ComputeLocalFrameAnglesXYZ(f3LocalX, f3LocalY, f3LocalZ, f3RayDir, fRayDirLocalZenith, fRayDirLocalAzimuth);
    f4LUTCoords.z = fRayDirLocalZenith;
    f4LUTCoords.w = fRayDirLocalAzimuth;

    f4LUTCoords.xyzw = f4LUTCoords.xyzw / float4(PI, 2*PI, PI/2, 2*PI) + float4(0.0, 0.5, 0, 0.5);

    // Clamp only zenith (yz) coordinate as azimuth is filtered with wraparound mode
    f4LUTCoords.xz = clamp(f4LUTCoords, 0.5/OPTICAL_DEPTH_LUT_DIM, 1.0-0.5/OPTICAL_DEPTH_LUT_DIM).xz;
}

void OpticalDepthLUTCoordsToWorldParams(in float4 f4LUTCoords, out float3 f3NormalizedStartPos, out float3 f3RayDir)
{
    float fStartPosZenithAngle  = f4LUTCoords.x * PI;
    float fStartPosAzimuthAngle = (f4LUTCoords.y - 0.5) * 2 * PI;
    f3NormalizedStartPos = ZenithAzimuthAngleToDirectionXZY(fStartPosZenithAngle, fStartPosAzimuthAngle);

    // Construct local tangent frame (z up)
    float3 f3LocalX, f3LocalY, f3LocalZ;
    ConstructLocalFrameXYZ(-f3NormalizedStartPos, float3(0,1,0), f3LocalX, f3LocalY, f3LocalZ);

    float fDirZentihAngle = f4LUTCoords.z * PI/2;
    float fDirLocalAzimuthAngle = (f4LUTCoords.w - 0.5) * 2 * PI;
    f3RayDir = GetDirectionInLocalFrameXYZ(f3LocalX, f3LocalY, f3LocalZ, fDirZentihAngle, fDirLocalAzimuthAngle);
}

float GetCloudRingWorldStep(uint uiRing, SGlobalCloudAttribs g_GlobalCloudAttribs)
{
    const float fLargestRingSize = g_GlobalCloudAttribs.fParticleCutOffDist * 2;
    uint uiRingDimension = g_GlobalCloudAttribs.uiRingDimension;
    uint uiNumRings = g_GlobalCloudAttribs.uiNumRings;
    float fRingWorldStep = fLargestRingSize / (float)((uiRingDimension) << ((uiNumRings-1) - uiRing));
    return fRingWorldStep;
}

float GetParticleSize(in float fRingWorldStep)
{
    return fRingWorldStep;
}

void ParticleScatteringLUTToWorldParams(in float4 f4LUTCoords, 
                                        out float3 f3StartPosUSSpace,
                                        out float3 f3ViewDirUSSpace,
                                        out float3 f3LightDirUSSpace,
                                        in uniform bool bSurfaceOnly)
{
    f3LightDirUSSpace = float3(0,0,1);
    float fStartPosZenithAngle = f4LUTCoords.x * PI;
    f3StartPosUSSpace = float3(0,0,0);
    sincos(fStartPosZenithAngle, f3StartPosUSSpace.x, f3StartPosUSSpace.z);

    float3 f3LocalX, f3LocalY, f3LocalZ;
    ConstructLocalFrameXYZ(-f3StartPosUSSpace, f3LightDirUSSpace, f3LocalX, f3LocalY, f3LocalZ);
    
    if( !bSurfaceOnly )
    {
        float fDistFromCenter = f4LUTCoords.w;
        // Scale the start position according to the distance from center
        f3StartPosUSSpace *= fDistFromCenter;
    }

    float fViewDirLocalAzimuth = (f4LUTCoords.y - 0.5) * (2 * PI); 
    float fViewDirLocalZenith = f4LUTCoords.z * ( bSurfaceOnly ? (PI/2) : PI );
    f3ViewDirUSSpace = GetDirectionInLocalFrameXYZ(f3LocalX, f3LocalY, f3LocalZ, fViewDirLocalZenith, fViewDirLocalAzimuth);
}

// All parameters must be defined in the unit sphere (US) space
float4 WorldParamsToParticleScatteringLUT(in float3 f3StartPosUSSpace, 
                                          in float3 f3ViewDirInUSSpace, 
                                          in float3 f3LightDirInUSSpace,
                                          in uniform bool bSurfaceOnly)
{
    float4 f4LUTCoords = 0;

    float fDistFromCenter = 0;
    if( !bSurfaceOnly )
    {
        // Compute distance from center and normalize start position
        fDistFromCenter = length(f3StartPosUSSpace);
        f3StartPosUSSpace /= max(fDistFromCenter, 1e-5);
    }
    float fStartPosZenithCos = dot(f3StartPosUSSpace, f3LightDirInUSSpace);
    f4LUTCoords.x = acos(fStartPosZenithCos);

    float3 f3LocalX, f3LocalY, f3LocalZ;
    ConstructLocalFrameXYZ(-f3StartPosUSSpace, f3LightDirInUSSpace, f3LocalX, f3LocalY, f3LocalZ);

    float fViewDirLocalZenith, fViewDirLocalAzimuth;
    ComputeLocalFrameAnglesXYZ(f3LocalX, f3LocalY, f3LocalZ, f3ViewDirInUSSpace, fViewDirLocalZenith, fViewDirLocalAzimuth);
    f4LUTCoords.y = fViewDirLocalAzimuth;
    f4LUTCoords.z = fViewDirLocalZenith;
    
    // In case the parameterization is performed for the sphere surface, the allowable range for the 
    // view direction zenith angle is [0, PI/2] since the ray should always be directed into the sphere.
    // Otherwise the range is whole [0, PI]
    f4LUTCoords.xyz = f4LUTCoords.xyz / float3(PI, 2*PI, bSurfaceOnly ? (PI/2) : PI) + float3(0, 0.5, 0);
    if( bSurfaceOnly )
        f4LUTCoords.w = 0;
    else
        f4LUTCoords.w = fDistFromCenter;
    if( bSurfaceOnly )
        f4LUTCoords.xz = clamp(f4LUTCoords.xyz, 0.5/SRF_SCATTERING_IN_PARTICLE_LUT_DIM, 1-0.5/SRF_SCATTERING_IN_PARTICLE_LUT_DIM).xz;
    else
        f4LUTCoords.xzw = clamp(f4LUTCoords, 0.5/VOL_SCATTERING_IN_PARTICLE_LUT_DIM, 1-0.5/VOL_SCATTERING_IN_PARTICLE_LUT_DIM).xzw;

    return f4LUTCoords;
}


#define SAMPLE_4D_LUT(tex3DLUT, LUT_DIM, f4LUTCoords, fLOD, Result)  \
{                                                               \
    float3 f3UVW;                                               \
    f3UVW.xy = f4LUTCoords.xy;                                  \
    float fQSlice = f4LUTCoords.w * LUT_DIM.w - 0.5;            \
    float fQ0Slice = floor(fQSlice);                            \
    float fQWeight = fQSlice - fQ0Slice;                        \
                                                                \
    f3UVW.z = (fQ0Slice + f4LUTCoords.z) / LUT_DIM.w;           \
                                                                \
    Result = lerp(                                              \
        tex3DLUT.SampleLevel(samLinearWrap, f3UVW, fLOD),       \
        /* frac() assures wraparound filtering of w coordinate*/                            \
        tex3DLUT.SampleLevel(samLinearWrap, frac(f3UVW + float3(0,0,1/LUT_DIM.w)), fLOD),   \
        fQWeight);                                                                          \
}

float HGPhaseFunc(float fCosTheta, const float g = 0.9)
{
    return (1/(4*PI) * (1 - g*g)) / pow( max((1 + g*g) - (2*g)*fCosTheta,0), 3.f/2.f);
}