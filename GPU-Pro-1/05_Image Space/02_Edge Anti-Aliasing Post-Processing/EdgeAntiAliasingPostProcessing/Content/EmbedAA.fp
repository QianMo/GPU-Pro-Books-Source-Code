varying vec2 Out_baseTexCoord;
varying vec3 Out_wNormal;
varying vec4 Out_DbgVal;
varying vec4 Out_SilVal;
varying vec2 Out_hNormalXY;
varying vec4 Out_Color;

uniform sampler2D textureMap;

vec3 frac(vec3 x)	{	return x - floor(x);	}
float saturate(float v)	{	return clamp(v, 0.0, 1.0);	}

float sign(float v)	{	if(v<0.0)	return -1.0;	else return +1.0;	}
vec2 sign(vec2 v)	{	return vec2(sign(v.x), sign(v.y));	}

void main(void)
{
	// BEGIN GL SPECIFIC
	vec2 baseTexCoord=Out_baseTexCoord;
	float silVal=Out_SilVal.r;
	vec2 hNormalXY=Out_hNormalXY;
	float dbgVal=0.0;
	// END GL SPECIFIC

    vec4 rc = vec4(1.0, 1.0, 1.0, 1.0);
    if(true)
    {
		float silhouetteParameter=silVal;
		float derivX=dFdx(silhouetteParameter);
		float derivY=dFdy(silhouetteParameter);
		vec2 approxEdgeDist=vec2(-silhouetteParameter/derivX, -silhouetteParameter/derivY);
		float coverage=min(abs(approxEdgeDist.x), abs(approxEdgeDist.y));
		
		dbgVal=coverage;
		// Encode these values
		//vec2 postStep=vec2(sign(approxEdgeDist.x), sign(approxEdgeDist.y));
		vec2 postStep=sign(hNormalXY)*vec2(+1.0,-1.0);
		float encodedPostStepX=(postStep.x>=0.0) ? 128.0 : 0.0;
		float encodedPostStepY=(postStep.y>=0.0) ? 64.0 : 0.0;
		float encodedVal=encodedPostStepX+encodedPostStepY+(saturate(coverage)*63.0);
		//rc.rgb=vec3(1,1,0);
		rc.rgb=texture2D( textureMap, baseTexCoord.xy ).rgb;
		rc.a=encodedVal/255.0;
		if(false)
		{	// Show postprocess codes
			rc.r=coverage;
			rc.g=(128.0+sign(postStep.x)*64.0)/255.0;
			rc.b=(128.0+sign(postStep.y)*64.0)/255.0;
		}	
		if(false)	//(coverage<0.01)
		{	// Indicate failure triangles
			rc.rgb=vec3(0.0,1.0,0.0);
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
	//rc.rgb=texture2D( textureMap, baseTexCoord.xy ).rgb;
#endif // ENABLE_UPRES_DEMO_EXTRAPOLATE
    }
   
	// BEGIN GL SPECIFIC
if(silVal<=0.0)
{
	rc.r=1.0;	gl_FragColor.rgb=vec3(0.0, 0.0, 0.0);	// DEVHACK

	discard(true);	
}

//rc.rgb=frac(baseTexCoord.x*vec3(0.01, 0.017, 0.4142));
	gl_FragColor=rc;
	// END GL SPECIFIC
}
