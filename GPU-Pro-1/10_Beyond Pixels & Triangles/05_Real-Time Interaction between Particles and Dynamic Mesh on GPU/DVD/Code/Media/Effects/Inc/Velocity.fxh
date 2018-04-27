shared cbuffer Velocity
{
	float4x4 matWVP;
	float4x4 matPrevWVP;

#if 0
	float4x4 matWV;

	float3 gRotAxis;
	float gAngularVelocity;
	float3 gLinearVelocity;
#endif
};


#if 0
float3 CalculateVel( float3 pos, float3 rotAxis, float3 lVel, float3 rVel )
{

	float l = length( pos ) + 0.000001;

	float3 rotDir = cross( pos / l, rotAxis );

	float3 vel = rVel + gAngularVelocity * l * rotDir;

	return vel;
}

float2 CalculateViewVel( float3 pos, float3 rotAxis, float3 lVel, float3 rVel )
{
	return mul( CalculateVel(pos, rotAxis, lVel, rVel), (float3x3)matWV ).xy;
}
#endif