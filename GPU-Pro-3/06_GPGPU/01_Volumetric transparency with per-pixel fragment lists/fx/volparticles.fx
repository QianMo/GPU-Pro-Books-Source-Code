Texture2D<float4> particleTexture;	//< Texture holding particle data (every triple is a particle: {position, radius}, {g, tau}, {orientation})
TextureCube puffTexture;			//< Distance impostor cube texture

float particleRadiusScale <
	string SasUiControl = "Slider";
	string SasUiLabel = "Particle radius scale";
	float SasUiMin = 0.1;
	float SasUiMax = 2;
	int SasUiSteps = 100;
>  = 1;
float particleDensityScale <
	string SasUiControl = "Slider";
	string SasUiLabel = "Particle density scale";
	float SasUiMin = 0.1;
	float SasUiMax = 2;
	int SasUiSteps = 100;
>  = 1;
float impostorScale <
	string SasUiControl = "Slider";
	string SasUiLabel = "Impostor scale";
	float SasUiMin = 0;
	float SasUiMax = 100;
	int SasUiSteps = 100;
>  = 40;
int displayedParticleCount
<
	string SasUiControl = "Slider";
	string SasUiLabel = "Particle count";
	float SasUiMin = 10;
	float SasUiMax = 300;
	int SasUiSteps = 290;
>  = 100;

cbuffer tileParticleCountBuffer
{
	uint nTileParticles[256];		//< Lists counts of particles visible in the 16x16 viewport tiles
}

/// Vertex shader output structure
struct VsosTile
{
	float4 pos				: SV_POSITION;
	float2 tex				: TEXCOORD0;
	float3 viewDir			: TEXCOORD1;
	float tileId			: TEXCOORD2;
};

/// Vertex shader to render 16x16 viewport tiles with instancing
VsosTile vsTile(IaosQuad input, in uint tileId : SV_InstanceId)
{
	VsosTile output = (VsosTile)0;

	// Find tile NDC coordinates
	input.pos.x += ((tileId % 16) * 2 - 15.0);
	input.pos.y += ((tileId / 16) * 2 - 15.0);
	input.pos.xy /= 16.0;
	output.pos = input.pos;

	// Find view direction
    float4 hWorldPosMinusEye = mul(input.pos, viewDirMatrix);
    hWorldPosMinusEye /= hWorldPosMinusEye.w;
    output.viewDir = hWorldPosMinusEye.xyz;
    output.pos.z = 0.99999;

	// Find tile texture coordinates
    output.tex = input.tex / 16;
	output.tex.x += (tileId % 16) / 16.0;
	output.tex.y += 1-(tileId / 16) / 16.0 - 1/16.0;
    output.tileId = tileId;
    return output;
};

/// Ray-sphere intersection routine
/// @param sphere_centre_r sphere centre position in xyz channels, radius in w channel
/// @param dir ray direction (ray origin is uniform, it is always eyePos)
/// @param maxDist maximum distance
/// @param enter_exit output result: intersected segment start and end point parameters in x and y channels, clipped to [0, maxDist]
/// @return enter_exit true if there is an intersection, false otherwise

bool intersectParticle(float4 sphere_centre_r, float3 dir, out float2 enter_exit, float maxDist)
{
	float3 centre = sphere_centre_r.xyz - eyePos;
	float ox = dot(centre, dir);
	float d2 = dot(centre, centre) - ox*ox;
	float radius = sphere_centre_r.w;
	float pm2 = radius * radius - d2;
	if(pm2 < 0)
		return false;
	float pm = sqrt(pm2);
	enter_exit = float2(ox - pm, ox + pm);
	enter_exit = max(0, enter_exit);
	enter_exit = min(enter_exit, maxDist);
	if(enter_exit.y == 0 || enter_exit.x == maxDist)
		return false;
	return true;

}

/// Quaternion multiplication
float4 quatMult(float4 a, float4 b)
{
	return float4(a.x * b.x - dot(a.yzw, b.yzw), cross(a.yzw, b.yzw) + a.x * b.yzw + b.x * a.yzw);
}

/// Pixel shader
float4 psParticles(VsosTile input) : SV_TARGET
{
	float3 dir = normalize(input.viewDir);

	// fetch opaque color and distance
	float4 opaque = opaqueTexture.SampleLevel(linearSampler, input.tex, 0);

	// initialize entry/exit list array
	int n = 1;
	float3 s[32];	// x: intersection dist, y: particle index (asfloat), z: delta factor (1/-1, entry/exit)
	s[0] = float3(opaque.w, 0, 0);	// opaque surface distance
	s[1] = float3(0, 0, 0);			// eye

	// initalize running variables
	float4 g_tau = 0;	// xyz: source term, w: extinction coeff
	float4 L_T = float4(0, 0, 0, 1); // xyz: L, w: T

	[loop] for(uint iParticle=0; iParticle < nTileParticles[input.tileId]; iParticle++)
	{
		float4 particle_centre_r = particleTexture.Load( uint3(iParticle*3, input.tileId, 0) );
		particle_centre_r.w *= particleRadiusScale;
		float2 entry_exit; // x: entry intersection distance, y: exit intersection distance
		if(intersectParticle(particle_centre_r, dir, entry_exit, opaque.w))
		{
			// compute point on sphere for distance impostor
			float3 gdir = dir * entry_exit.x + eyePos - particle_centre_r.xyz;
			// rotate direction with particle orientation quaternion
			float4 quat = particleTexture.Load( uint3(iParticle*3+2, input.tileId, 0) );
			gdir = quatMult(quatMult(quat, float4(gdir, 0)), quat*float4(1, 1, 1, -1));

			// modify sphere radius with impostor value
			float texe = saturate(1-puffTexture.SampleLevel(linearSampler, gdir, 0)) * impostorScale;
			particle_centre_r.w -= texe;
			// re-execute intersection calculation
			if(particle_centre_r.w > 0 && intersectParticle(particle_centre_r, dir, entry_exit, opaque.w)) 
			{
				// compute artificial edge attenuation factor			
				float deltaFactor = saturate((entry_exit.y - entry_exit.x) * 0.004 );
				deltaFactor *= deltaFactor * 6 * particleDensityScale;

				// insert exit point into array, moving elements
				int iDest = n + 2;
				int iSrc = n;
				n = iDest;
				[loop] while(iSrc >= 0 && s[iSrc].x < entry_exit.x)
				{
					s[iDest] = s[iSrc];
					iDest--;
					iSrc--;
				}
				s[iDest] = float3(entry_exit.x, asfloat(iParticle*3+1), deltaFactor );
				iDest--;
				// insert entry point into array, moving elements
				[loop] while(iSrc >= 0 && s[iSrc].x < entry_exit.y)
				{
					s[iDest] = s[iSrc];
					iDest--;
					iSrc--;
				}
				s[iDest] = float3(entry_exit.y, asfloat(iParticle*3+1), -deltaFactor );
			}
	
			// process entry/exit points in the safe zone	
			float safeDist = length(particle_centre_r.xyz - eyePos) - particle_centre_r.w;
			[loop] while(n > 0 && (s[n-1].x < safeDist || n > 29))
			{
				g_tau += particleTexture.Load( uint3(asuint(s[n].y), input.tileId, 0) ) * s[n].z;
				float transparency = exp( - (s[n-1].x - s[n].x) * g_tau.w );
				if(g_tau.w > 0.000001)
					L_T.xyz += L_T.w * g_tau.xyz * (1 - transparency) /  g_tau.w;
				L_T.w *= transparency;
				n--;
			}
		}
	}
	// process remaining entry/exit points
	[loop] while(n > 0)
	{
		g_tau += particleTexture.Load( uint3(asuint(s[n].y), input.tileId, 0) ) * s[n].z;
		float transparency = exp( - (s[n-1].x - s[n].x) * g_tau.w );
		if(g_tau.w > 0.000001)
			L_T.xyz += L_T.w * g_tau.xyz * (1 - transparency) /  g_tau.w;
		L_T.w *= transparency;
		n--;
	}
	return L_T + L_T.w * opaque;
};

/// Pixel shader without distance impostors
float4 psSpheres(VsosTile input) : SV_TARGET
{
	float3 dir = normalize(input.viewDir);

	// fetch opaque color and distance
	float4 opaque = opaqueTexture.SampleLevel(linearSampler, input.tex, 0);

	// initialize entry/exit list array
	int n = 1;
	float3 s[32];	// x: intersection dist, y: particle index (asfloat), z: delta factor (1/-1, entry/exit)
	s[0] = float3(opaque.w, 0, 0);	// opaque surface distance
	s[1] = float3(0, 0, 0);			// eye

	// initalize running variables
	float4 g_tau = 0;	// xyz: source term, w: extinction coeff
	float4 L_T = float4(0, 0, 0, 1); // xyz: L, w: T

	[loop] for(uint iParticle=0; iParticle < nTileParticles[input.tileId]; iParticle++)
	{
		float4 particle_centre_r = particleTexture.Load( uint3(iParticle*3, input.tileId, 0) );
		particle_centre_r.w *= particleRadiusScale;
		float2 entry_exit; // x: entry intersection distance, y: exit intersection distance
		if(intersectParticle(particle_centre_r, dir, entry_exit, opaque.w))
		{
			float deltaFactor = particleDensityScale;
			// insert exit point into array, moving elements
			int iDest = n + 2;
			int iSrc = n;
			n = iDest;
			[loop] while(iSrc >= 0 && s[iSrc].x < entry_exit.x)
			{
				s[iDest] = s[iSrc];
				iDest--;
				iSrc--;
			}
			s[iDest] = float3(entry_exit.x, asfloat(iParticle*3+1), deltaFactor );
			iDest--;
			// insert entry point into array, moving elements
			[loop] while(iSrc >= 0 && s[iSrc].x < entry_exit.y)
			{
				s[iDest] = s[iSrc];
				iDest--;
				iSrc--;
			}
			s[iDest] = float3(entry_exit.y, asfloat(iParticle*3+1), -deltaFactor );
	
			// process entry/exit points in the safe zone	
			float safeDist = length(particle_centre_r.xyz - eyePos) - particle_centre_r.w;
			[loop] while(n > 0 && (s[n-1].x < safeDist || n > 29))
			{
				g_tau += particleTexture.Load( uint3(asuint(s[n].y), input.tileId, 0) ) * s[n].z;
				float transparency = exp( - (s[n-1].x - s[n].x) * g_tau.w );
				if(g_tau.w > 0.000001)
					L_T.xyz += L_T.w * g_tau.xyz * (1 - transparency) /  g_tau.w;
				L_T.w *= transparency;
				n--;
			}
		}
	}
	// process remaining entry/exit points
	[loop] while(n > 0)
	{
		g_tau += particleTexture.Load( uint3(asuint(s[n].y), input.tileId, 0) ) * s[n].z;
		float transparency = exp( - (s[n-1].x - s[n].x) * g_tau.w );
		if(g_tau.w > 0.000001)
			L_T.xyz += L_T.w * g_tau.xyz * (1 - transparency) /  g_tau.w;
		L_T.w *= transparency;
		n--;
	}
	return L_T + L_T.w * opaque;
};

technique11 smoke
{
    pass smoke
    {
        SetVertexShader ( CompileShader( vs_5_0, vsTile() ) );
        SetGeometryShader( NULL );
        SetRasterizerState( defaultRasterizer );
        SetPixelShader( CompileShader( ps_5_0, psParticles() ) );
		SetDepthStencilState( defaultCompositor, 0);
		SetBlendState(defaultBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff);
    }
    pass smokeSpheres
    {
        SetVertexShader ( CompileShader( vs_5_0, vsTile() ) );
        SetGeometryShader( NULL );
        SetRasterizerState( defaultRasterizer );
        SetPixelShader( CompileShader( ps_5_0, psSpheres() ) );
		SetDepthStencilState( defaultCompositor, 0);
		SetBlendState(defaultBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff);
    }
}
