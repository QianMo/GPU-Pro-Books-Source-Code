// -----------------------------------------------------------------------------
// ---------------------------- ReliefConeMapping ------------------------------
// -----------------------------------------------------------------------------

struct ReliefConeMappingVertexShaderInput 
{
    float4 position	: POSITION;
    float3 normal	: NORMAL;
    float2 texCoord	: TEXCOORD0;
    float3 tangent	: TEXCOORD1;
    float4 binormal	: TEXCOORD2;
};

struct ReliefConeMappingVertexShaderOutput
{
	float4 position		: POSITION;
	float2 texCoord		: TEXCOORD0;
	float3 vertexPos	: TEXCOORD1;
	float3 normal		: TEXCOORD2;
	float3 tangent		: TEXCOORD3;
	float3 binormal		: TEXCOORD4;
	float3 light		: TEXCOORD5;
	float4 smTexCoord	: TEXCOORD6;
};

// Heighfield intersection methods
// RayIntersectRelaxedcone method uses relaxed cones
void RayIntersectRelaxedcone(sampler2D normalMap, inout float3 rayPos, inout float3 rayDir)
{
	const int coneSteps=15;
	const int binarySteps=8;
	
	rayDir /= rayDir.z;
	float rayRatio = length(rayDir.xy);
	float3 pos = rayPos;
	
	for (int i=0; i<coneSteps; i++)
	{
		float4 tex = tex2D(normalMap, rayPos.xy);
		float coneRatio = tex.z;
		float height = saturate(tex.w - rayPos.z);
		float d = coneRatio * height / (rayRatio + coneRatio);
		rayPos += rayDir * d;
	}

	rayDir *= rayPos.z*0.5;
	rayPos = pos + rayDir;

	for (int i=0; i<binarySteps; i++)
	{
		float4 tex = tex2D(normalMap, rayPos.xy);
		rayDir *= 0.5;
		if (rayPos.z<tex.w)
			rayPos+=rayDir;
		else
			rayPos-=rayDir;
	}
}

// RayIntersectReliefMap methdo uses normal intersection test (used for self shadow calculation)
float RayIntersectReliefMap(in sampler2D reliefmap, in float2 dp, in float2 ds)
{
	const int linearSteps = 15;
	const int binarySteps = 5;
	float depth_step = 1.0/linearSteps;

	float size = depth_step;
	float depth = 0.0f;
	float bestDepth = 1.0f;

	for (int i=0; i<linearSteps-1; i++)
	{
		depth += size;
		float4 t = tex2D(reliefmap, dp + ds*depth);

		if (bestDepth > 0.996)
			if (depth >= t.w)
				bestDepth = depth;
	}
	depth = bestDepth;
	
	for (int i=0; i<binarySteps; i++)
	{
		size *= 0.5f;
		float4 t = tex2D(reliefmap, dp + ds*depth);
		if (depth >= t.w)
		{
			bestDepth = depth;
			depth -= 2.0f*size;
		}
		depth+=size;
	}

	return bestDepth;
}

ReliefConeMappingVertexShaderOutput ReliefConeMappingVP(ReliefConeMappingVertexShaderInput IN)
{
	ReliefConeMappingVertexShaderOutput OUT;

	// Vertex position in object space
	float4 pos = float4(IN.position.xyz, 1.0f);

	// Vertex position in clip space
	OUT.position = mul(glstate.matrix.mvp, pos);
	OUT.texCoord = IN.texCoord;

	// Vertex position in view space (with model transformations)
	float3 vpos = mul(glstate.matrix.modelview[0], pos).xyz;
	OUT.vertexPos = vpos;

	// Compute modelview rotation only part
	float3x3 modelviewrot;
	modelviewrot[0] = glstate.matrix.modelview[0][0].xyz;
	modelviewrot[1] = glstate.matrix.modelview[0][1].xyz;
	modelviewrot[2] = glstate.matrix.modelview[0][2].xyz;

	// tangent space vectors in view space
	OUT.normal = mul(modelviewrot, IN.normal);
	OUT.tangent = mul(modelviewrot, IN.tangent);
	OUT.binormal = mul(modelviewrot, IN.binormal.xyz);
	
	// Light position in tangent space
	float3 light = mul(viewMatrix, float4(lightPosObjSpace.xyz, 1.0f)).xyz;
	//OUT.light = light-vpos;
	OUT.light = light;

	float zEyeSpace = mul(mul(shadowMapLinearTextureMatrix, glstate.matrix.modelview[0]), IN.position).z;
	OUT.smTexCoord = mul(mul(shadowMapTextureMatrix, glstate.matrix.modelview[0]), IN.position);
	OUT.smTexCoord.z = (zEyeSpace + clipPlanes.x) / clipPlanes.z;

	return OUT;
}

float4 NormalMappingFP(ReliefConeMappingVertexShaderOutput IN) : COLOR
{
	float shadowTerm = 1.0f;
	float2 uv = IN.smTexCoord.xy/IN.smTexCoord.w;

	if (useMipMaps)
		shadowTerm = ShadowTerm(csmSinMap, csmCosMap, float3(uv, IN.smTexCoord.z), log2(0.0075f*shadowMapSize));
	else
		shadowTerm = ShadowTerm(csmSinMap, csmCosMap, float3(uv, IN.smTexCoord.z), 0.0001f*shadowMapSize);

	//float2 dz_duv = DepthGradient(IN.smTexCoord.xy/IN.smTexCoord.w, IN.smTexCoord.z);
	//shadowTerm =  PCF_Filter(shadowMap, IN.smTexCoord, dz_duv, 0.005f);

	////////////////////////////////////////////////////////////////////////////////////////////////

	float4 normal = tex2D(normalMap, IN.texCoord);
	normal.xy = normal.xy*2.0f - 1.0f; // trafsform to [-1,1] range

	normal.xy = -normal.xy;

	normal.z = sqrt(1.0 - dot(normal.xy, normal.xy)); // compute z component

	// transform normal to world space
	normal.xyz = normalize(normal.x*IN.tangent - normal.y*IN.binormal + normal.z*IN.normal);

	// color map
	float4 color = tex2D(textureMap, IN.texCoord);

	// view and light directions
	float3 v = normalize(IN.vertexPos);
	float3 l = normalize(IN.light.xyz - IN.vertexPos);

	// compute diffuse and specular terms
	float att = saturate(dot(l, IN.normal.xyz));
	float diff = shadowTerm*saturate(dot(l, normal.xyz));
	float spec = saturate(dot(normalize(l - v), normal.xyz));

	// compute final color
	float4 finalcolor;
	//finalcolor.xyz = Ka*color.xyz + att*(color.xyz*Kd.xyz*diff + Ks.xyz*pow(spec, shininess));
	finalcolor.xyz = Ka*color.xyz + color.xyz*Kd.xyz*diff + Ks.xyz*pow(spec, shininess);
	//finalcolor.xyz = diff;
	finalcolor.w=1.0;

	return finalcolor;
}

FragmentShaderOutput ReliefConeMappingFP(ReliefConeMappingVertexShaderOutput IN)
{
	FragmentShaderOutput OUT;

	// TODO: add uniform parameter
	float depth = 0.1f;

	// Tangent space transformations
	float3x3 tangentSpace = float3x3(IN.tangent, IN.binormal, IN.normal);
	//float3x3 invTangentSpace = transpose(tangentSpace);

	float3 eye = normalize(mul(tangentSpace, IN.vertexPos));

	// Setup ray position and ray direction
	float3 rayPos, rayDir;

	rayPos = float3(IN.texCoord,0);
	rayDir = eye;
	rayDir.z = abs(rayDir.z);

	float db = 1.0-rayDir.z; db*=db; db*=db; db=1.0-db*db;
	rayDir.xy *= db;
	
	rayDir.xy *= depth;

	// Perform intersection test
	RayIntersectRelaxedcone(normalMap, rayPos, rayDir);

	// Texture color
	float4 color = tex2D(textureMap,rayPos.xy);
	
	// Normal map
	float4 normal = tex2D(normalMap,rayPos.xy);
	normal.xy = 2*normal.xy - 1;
	normal.z = sqrt(1.0 - dot(normal.xy,normal.xy));

	// Light and view in tangent space
	float3 l = normalize(mul(tangentSpace, IN.light));

	// Compute shadow with shadow map
	float shadow = 1.0;

	float3 position = IN.vertexPos;
	position.z += 1.0f - rayPos.z;

	// Transform pixel position to shadow map space
	//float4 sm = mul (shadowMapTextureMatrix, float4(position, 1.0f));   
	//sm /= sm.w; 
	//if (sm.z > tex2Dproj(shadowMap, sm).x)
	//{
	//	shadow=dot(Ka,1)*0.333333;
	//	Ks=0;
	//}

	// If fragment not already in shadow, compute self shadow
	if (shadow >= 1.0f)
	{
		float3 vT  = normalize(IN.vertexPos);
		float aT  = dot(IN.normal,-vT);

		// Compute light direction
		float3 lT = normalize(vT*rayPos.z*aT-IN.light.xyz);

		// Ray intersection in light direction
		float2 dp = rayPos.xy;
		float a  = dot(IN.normal,-lT);
		float3 s  = normalize(float3(dot(lT,IN.tangent),dot(lT,IN.binormal),a));
		s *= depth/a;
		float2 ds = s.xy;
		dp -= ds*rayPos.z;
		float dl = RayIntersectReliefMap(normalMap,dp,s.xy);
	
		if (dl<(rayPos.z-0.05f)) // If pixel in shadow
		{
			shadow=dot(Ka,1)*0.333333;
			Ks=0;
		}
	}

	// Compute diffuse and specular terms
	float diff = shadow*saturate(dot(l,normal.xyz));
	float spec = saturate(dot(normalize(l-eye),normal.xyz));

	// Attenuation factor
	float att = 1.0 - max(0,l.z); 
	att = 1.0 - att*att;
	
	// Compute final color
	OUT.color.xyz = Ka*color.xyz + att*(color.xyz*Kd.xyz*diff + Ks*pow(spec, shininess));
	OUT.color.w = 1.0f;

	return OUT;
}