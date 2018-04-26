#define ENABLE_ALLEDGEBLUR_DEMO
varying vec2 Out_baseTexCoord;
varying vec3 Out_wNormal;
varying vec4 Out_DbgVal;
varying vec4 Out_SilVal;
varying vec2 Out_hNormalXY;
varying vec4 Out_Color;
varying vec3 Out_BaryParam;

uniform sampler2D textureMap;

vec3 frac(vec3 x)	{	return x - floor(x);	}
float saturate(float v)	{	return clamp(v, 0.0, 1.0);	}

float sign(float v)	{	if(v<0.0)	return -1.0;	else return +1.0;	}
vec2 sign(vec2 v)	{	return vec2(sign(v.x), sign(v.y));	}

void main(void)
{
	// BEGIN GL SPECIFIC
	vec4 outColor=0;
	vec4 outEdgeHint=0;
	vec2 baseTexCoord=Out_baseTexCoord;
	float silVal=Out_SilVal.r;
	vec2 hNormalXY=Out_hNormalXY;
	vec3 baryParam=Out_BaryParam;
	float dbgVal=0.0;
	// END GL SPECIFIC

	outColor.rgb=texture2D( textureMap, baseTexCoord.xy ).rgb;

    vec4 rc = vec4(1.0, 1.0, 1.0, 1.0);
    if(true)
    {
		float silhouetteParameter=silVal;
		float derivX=dFdx(silhouetteParameter);
		float derivY=dFdy(silhouetteParameter);
		
		vec2 approxEdgeDist=vec2(-silhouetteParameter/derivX, -silhouetteParameter/derivY);
		float coverage=min(abs(approxEdgeDist.x), abs(approxEdgeDist.y));
		if((silhouetteParameter==0.0) || (silhouetteParameter==1.0))
		{	// Clean up problems when none or all three verts pass the adjoining-backface test
			coverage=1;
		}
		//vec2 postStep=sign(hNormalXY)*vec2(+1.0,-1.0);
		vec2 postStep=vec2(sign(approxEdgeDist.x), sign(approxEdgeDist.y))*vec2(+1.0,-1.0);
    
#ifdef ENABLE_ALLEDGEBLUR_DEMO
		// "Soft" here has the same meaning as "half-open" - the sense is that 
		//	we're providing a softer blend.
		// Possibly override coverage, approxedgedist, poststep
		if((coverage>=1) || (coverage==0))
		{
			vec3 triV=1-abs(baryParam);
			float minV=1.0;
			float minV_DX=0.0;
			float minV_DY=0.0;
			float rawBaryParam=0.0;
			if(triV.x<minV)	{	minV=triV.x;	minV_DX=dFdx(triV.x);	minV_DY=dFdy(triV.x);	rawBaryParam=baryParam.x;	}
			if(triV.y<minV)	{	minV=triV.y;	minV_DX=dFdx(triV.y);	minV_DY=dFdy(triV.y);	rawBaryParam=baryParam.y;	}
			if(triV.z<minV)	{	minV=triV.z;	minV_DX=dFdx(triV.z);	minV_DY=dFdy(triV.z);	rawBaryParam=baryParam.z;	}
			
			vec2 approxEdgeDistSoft=vec2(-minV/minV_DX, -minV/minV_DY);
			float coverageSoft=min(abs(approxEdgeDistSoft.x), abs(approxEdgeDistSoft.y));
//outColor.rgb=coverageSoft*0.15;
						
			if(coverageSoft<=1.0)
			{
				// postStep can't be calculated using the interpolated surface normal, since
				//	the distance-to-edge could be for a non-silhouette edge.
				//	We use the rate of change of the silhouette parameter instead.
				postStep=vec2(sign(approxEdgeDistSoft.x), sign(approxEdgeDistSoft.y))*vec2(+1.0,-1.0);
				coverage=coverageSoft;
				if(rawBaryParam<0.0)
				{	// This edge is half-open; blur accordingly.
					coverage=coverageSoft+0.5;
				}
			}
		}
#endif // ENABLE_ALLEDGEBLUR_DEMO
//outColor.rgb=coverage;

		// Encode these values
		float encodedPostStepX=(postStep.x>=0.0) ? 128.0 : 0.0;
		float encodedPostStepY=(postStep.y>=0.0) ? 64.0 : 0.0;
		float encodedVal=encodedPostStepX+encodedPostStepY+(saturate(coverage)*63.0);
		outColor.a=encodedVal/255.0;
		if(false)
		{	// Show postprocess codes
			outColor.r=coverage;
			outColor.g=(128.0+sign(postStep.x)*64.0)/255.0;
			outColor.b=(128.0+sign(postStep.y)*64.0)/255.0;
		}	
		
#ifdef ENABLE_SHADOWMAP_DEMO_EXTRAPOLATE
	vec2 offsetToEncode=vec2(0.0, 0.0);
	float encWeight=0.0;
	if(silhouetteParameter<0.90)
	{
		float d=sqrt((derivX*derivX)+(derivY*derivY))*1.0;
		offsetToEncode.x=derivX/d;
		offsetToEncode.y=derivY/d;
		encWeight=silhouetteParameter/d;
	}
	rc.r=saturate(0.5 + offsetToEncode.x*0.5);
	rc.g=saturate(0.5 + offsetToEncode.y*0.5);
	rc.b=encWeight;
	rc.a=0.3;
#endif // ENABLE_SHADOWMAP_DEMO_EXTRAPOLATE

#ifdef ENABLE_SHADOWMAP_DEMO_BORDERSEARCH
	rc=0;
	rc.rg=coverage*0.5;
	rc.a=0.3;
#endif // ENABLE_SHADOWMAP_DEMO_BORDERSEARCH

#ifdef ENABLE_UPRES_DEMO_EXTRAPOLATE
	vec2 offsetToEncode=vec2(0.0, 0.0);
	float encWeight=1.0;
	if(silhouetteParameter<0.90)
	{
		float d=sqrt((derivX*derivX)+(derivY*derivY))*1.0;
		float normVal=silhouetteParameter/d;
		if(normVal<1.0)
		{
			offsetToEncode.x=derivX/d;
			offsetToEncode.y=derivY/d;
			encWeight=silhouetteParameter/d;
		}
	}
	outEdgeHint.r=saturate(0.5 + offsetToEncode.x*0.5);
	outEdgeHint.g=saturate(0.5 + offsetToEncode.y*0.5);
	outEdgeHint.b=encWeight;
	outEdgeHint.a=0.3;
	outColor.rgb=texture2D( textureMap, baseTexCoord.xy, -14.0 ).rgb;
#endif // ENABLE_UPRES_DEMO_EXTRAPOLATE

//outColor.a=saturate(silhouetteParameter)*0.9;
//outColor.a=saturate(silVal)*0.9;
//outColor.rgb=saturate(silhouetteParameter)*0.9;
//outColor.rgb=saturate(coverage)*0.9;
//outColor.rgb=Out_wNormal*0.5+0.5;
    }
   
	// BEGIN GL SPECIFIC
if(silVal<=0.0)
{
	//outColor.r=1.0;		//outColor.rgb=vec3(0.0, 0.0, 0.0);	// DEVHACK
	//discard(true);	
}

//outColor.rgb=Out_DbgVal.rgb;	// DEVHACK
//outColor.rgb=silVal;
	gl_FragData[0]=outColor;
	gl_FragData[1]=outEdgeHint;
	// END GL SPECIFIC
}
