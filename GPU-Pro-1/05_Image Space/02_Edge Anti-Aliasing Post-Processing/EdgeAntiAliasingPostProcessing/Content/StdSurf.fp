varying vec3 wNormal;
varying vec2 texCoord;
varying vec3 wPos;
varying vec3 wSpecHalfAngle;

// Illum values in Max coords
varying vec3 lColBase;
varying vec3 lColX;
varying vec3 lColY;
varying vec3 lColZ;

varying vec3 wBasisUInterp;

uniform sampler2D texLighting;
uniform sampler2D texDiffuse;
uniform sampler2D texNormal;
uniform sampler2D texShadow;
uniform sampler2D texComplete;
uniform sampler2D texExNormal;
uniform sampler2D texExDiffuse;

float frac(float x)	{	return x - floor(x);	}
vec3 frac(vec3 v)	{	return v - floor(v);	}
float saturate(float x)	{	if(x<0.0)	return 0.0;	else if(x>1.0)	return 1.0; else return x;	}
vec3 saturate(vec3 v)	{	return vec3(saturate(v.x), saturate(v.y), saturate(v.z));	}

vec3 lerp(vec3 a, vec3 b, float p)
{
	return (a*(1.0-p)) + (b*p);
}
vec3 LinearToGamma(vec3 c)
{
	return vec3(
		pow(c.r, 1.0/2.2),
		pow(c.g, 1.0/2.2),
		pow(c.b, 1.0/2.2)	);
}

vec3 MaxToGD(vec3 v)
{
	return v.yzx*vec3(-1.0,1.0,1.0);
}

void main(void)
{
	vec3 colNeutral=vec3(0.5, 0.5, 0.5);
	vec3 colRedBright=vec3(1.0, 0.0, 0.0);
	vec3 colRedDark=vec3(0.2, 0.0, 0.0);

	vec3 wSurfaceNormal=wNormal;
	vec3 wBasisU=normalize(wBasisUInterp);
	vec3 wBasisV=normalize(cross(wNormal, wBasisU));

	vec3 colComplete=texture2D(texComplete, texCoord).rgb;
	vec3 colDiffuse=texture2D(texDiffuse, texCoord).rgb;
	vec3 colNormal=texture2D(texNormal, texCoord).rgb;
	vec3 colShadow=texture2D(texShadow, texCoord).rgb;
	vec3 colExNormal=texture2D(texExNormal, texCoord*12.0).rgb;
	vec3 colExDiffuse=texture2D(texExDiffuse, texCoord*12.0).rgb;
//colDiffuse=colExDiffuse;

vec2 transMap=texCoord*50;
vec2 localVec=transMap-floor(transMap);
vec3 localNormal=normalize(vec3(localVec.x-0.5, localVec.y-0.5, 1));
localNormal=normalize(colExNormal.rgb-vec3(0.5, 0.5, 0.5));
vec3 wTestNormal=(wBasisU*localNormal.x)+(wBasisV*localNormal.y)+(wNormal*localNormal.z);

	// Calculate surface normal
	wSurfaceNormal=(colNormal.yzx-vec3(0.5,0.5,0.5))*vec3(-1.0, 1.0, 1.0)*2.0;	// Matched to wNormal
	wSurfaceNormal=normalize(wSurfaceNormal);
//wSurfaceNormal=normalize(wTestNormal);

	// BEGIN CALC INDIRECTLIGHT
	// Calculate incident light; replicates cPackedLightParam::Evaluate()
	vec3 indirectLight=lColBase+
		(wSurfaceNormal.x*-lColY)+
		(wSurfaceNormal.y*lColZ)+
		(wSurfaceNormal.z*lColX);
	// END CALC INDIRECTLIGHT

	vec3 lightDir, lDirect, specCol;
	float shadowFactor=colShadow.r;
	if(false)
	{	// Old version - for TestSurf
		lightDir=-normalize(vec3(466.588,248.193,-407.375));
		lDirect=vec3(0.62,0.59,0.47)*3.5;
		specCol=vec3(1,1,1);
	}
	else
	{
		lightDir=normalize(vec3(0.9, 0.4, 1.0));
		lDirect=vec3(0.62,0.59,0.47)*10.0;
		//shadowFactor=0;
		specCol=vec3(1,1,1);
	}
	// BEGIN CALC DIRECTLIGHT
	// Diffuse
	float dprDirect=saturate(dot(MaxToGD(lightDir), wSurfaceNormal));
	vec3 diffuseLight=(dprDirect)*lDirect;
	// Specular
	float dprSpec=saturate(dot(wSurfaceNormal, normalize(wSpecHalfAngle)));
	//dprSpec=saturate(dot(wSurfaceNormal, MaxToGD(lightDir)));	// DEV: Match with colComplete
	vec3 specLight=specCol*pow(dprSpec, 8.0)*1.0;
	vec3 directLight=(diffuseLight+specLight)*shadowFactor;
	// END CALC DIRECTLIGHT

	vec3 litCol=(indirectLight+directLight)*colDiffuse;
litCol=indirectLight*colDiffuse*shadowFactor;
	//gl_FragColor.rgb=pow(saturate((litCol-0.05)*1.5), 0.8);
	gl_FragColor.rgb=pow(litCol*2.0, 1.0/2.2);
	gl_FragColor.a=1;

	//gl_FragColor.rgb=texCoord.xyy;
	//gl_FragColor.rgb=directLight;
	//gl_FragColor.rgb=colNormal;
	//gl_FragColor.rgb=colShadow;
	//gl_FragColor.rgb=lColBase;
	//gl_FragColor.rgb=indirectLight;
	//gl_FragColor.rgb=colComplete;
	//gl_FragColor.rgb=LinearToGamma(gl_FragColor.rgb);

	//gl_FragColor.rgb=((wSurfaceNormal*0.5)+vec3(0.5,0.5,0.5));
	//gl_FragColor.rgb=colNormal;
	//gl_FragColor.rgb=(wNormal*0.5)+vec3(0.5,0.5,0.5);
	//gl_FragColor.rgb=(wBasisUInterp*0.5)+vec3(0.5,0.5,0.5);

	//gl_FragColor.rgb=lColBase;
	//gl_FragColor.rgb=normalize(lDirKey)*0.5+vec3(0.5,0.5,0.5);
	//gl_FragColor.rgb=diffuseLight;
	//gl_FragColor.rgb=vec3(1,1,1)*dprSpec;
	//gl_FragColor.rgb=specLight;
	//gl_FragColor.rgb=texture2D(texComplete, texCoord).rgb;
	//gl_FragColor.rgb=texture2D(texLighting, texCoord).rgb;
	//gl_FragColor.rgb=colDiffuse.rgb;
	//gl_FragColor.rgb=colLitIndirect.rgb;
	gl_FragColor.rgb=pow(indirectLight*colDiffuse*1.5, 1.0/1.6);
	//gl_FragColor.rgb=pow(litCol, 1.0/2.2);
	//gl_FragColor.rgb=shadowFactor;


	/*vec2 deriv;
	deriv.x=dFdx(texCoord.x);
	deriv.y=dFdx(texCoord.y);
	vec2 convDeriv=(normalize(deriv)*0.5)+0.5;
	gl_FragColor.rgb=convDeriv.xxy;*/
}
