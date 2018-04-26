varying vec2 texCoord;
uniform sampler2D texColor, texEdgeHint;
const int maxRenderSize=1024;

vec2 frac(vec2 x)	{	return x - floor(x);	}
vec3 frac(vec3 x)	{	return x - floor(x);	}
vec2 lerp(float a, float b, float s)	{	return (a*(1.0-s)) + (b*s);	}
vec2 lerp(vec2 a, vec2 b, float s)	{	return (a*(1.0-s)) + (b*s);	}
vec3 lerp(vec3 a, vec3 b, float s)	{	return (a*(1.0-s)) + (b*s);	}
float saturate(float v)	{	return clamp(v, 0.0, 1.0);	}
vec2 saturate(vec2 v)	{	return vec2(
	clamp(v.x, 0.0, 1.0),
	clamp(v.y, 0.0, 1.0));	}
vec3 saturate(vec3 v)	{	return vec3(
	clamp(v.x, 0.0, 1.0),
	clamp(v.y, 0.0, 1.0),
	clamp(v.z, 0.0, 1.0));	}
vec4 saturate(vec4 v)	{	return vec4(
	clamp(v.x, 0.0, 1.0),
	clamp(v.y, 0.0, 1.0),
	clamp(v.z, 0.0, 1.0),
	clamp(v.w, 0.0, 1.0));	}

vec4 tex2DOffset(vec2 uv, vec2 offset)
{
	float texSize=float(maxRenderSize);	//*0.25;
	return texture2D(texColor, uv+(vec2(+1, -1)*offset/texSize));
}
vec3 CalcWeightFunction(vec2 samplePt, vec3 sampleCol, float fallMag)
{
	vec2 unpackedDist=(sampleCol.rg-0.5)*2.0;
	float baseWeight=sampleCol.b;
	vec3 rc=0;
	rc.x=unpackedDist.x;
	rc.y=unpackedDist.y;
	rc.z=baseWeight;
	// Tilt slightly down
	vec2 fallVec=vec2(1,1)-(samplePt*2);
	rc.xy+=fallVec*-fallMag;	//rc.xy+=fallVec*-0.02;
	rc.z-=(rc.x*samplePt.x)+(rc.y*samplePt.y);
	return rc;
}

//#ifdef ENABLE_SHADOWMAP_DEMO_EXTRAPOLATE

vec3 GetShadowMapSample_Extrapolate(vec2 uv)
{
	vec3 rc=0.0;
	vec2 expUV=uv*float(maxRenderSize);
	vec2 fracUV=frac(expUV);
	vec2 baseUV=((expUV-fracUV)+0.5)/float(maxRenderSize);
	vec2 singleUVStep=1.0/float(maxRenderSize);
	
	vec4 colAA=texture2D(texEdgeHint, baseUV + (vec2(0,0)*singleUVStep) );
	vec4 colBA=texture2D(texEdgeHint, baseUV + (vec2(1,0)*singleUVStep) );
	vec4 colAB=texture2D(texEdgeHint, baseUV + (vec2(0,1)*singleUVStep) );
	vec4 colBB=texture2D(texEdgeHint, baseUV + (vec2(1,1)*singleUVStep) );
	
	vec3 weightFuncAA=CalcWeightFunction(vec2(0,0), colAA, 0.02);
	vec3 weightFuncBA=CalcWeightFunction(vec2(1,0), colBA, 0.02);
	vec3 weightFuncAB=CalcWeightFunction(vec2(0,1), colAB, 0.02);
	vec3 weightFuncBB=CalcWeightFunction(vec2(1,1), colBB, 0.02); 
	vec3 fracPosH=vec3(fracUV.x, fracUV.y, 1.0);
	float weightAA=dot(weightFuncAA, fracPosH);
	float weightBA=dot(weightFuncBA, fracPosH);
	float weightAB=dot(weightFuncAB, fracPosH);
	float weightBB=dot(weightFuncBB, fracPosH);
	
	float maxW=max(max(weightAA, weightBA), max(weightAB, weightBB));
	vec4 offsetW=maxW-vec4(weightAA, weightBA, weightAB, weightBB);
	float transDist=1.0/18.0;
	vec4 satW=saturate(transDist-offsetW)/transDist;
	float netW=dot(satW, vec4(1,1,1,1));
	vec4 vecA=vec4(colAA.a, colBA.a, colAB.a, colBB.a);
	float combA=dot(vecA, satW)/netW;

	float w=-10.0;
	float a=0.0;
	if(weightAA>w)	{	w=weightAA;	a=colAA.a;	}
	if(weightBA>w)	{	w=weightBA;	a=colBA.a;	}
	if(weightAB>w)	{	w=weightAB;	a=colAB.a;	}
	if(weightBB>w)	{	w=weightBB;	a=colBB.a;	}
//a=weightFuncAB.y;
a=combA;
//return colAA.rgb;
	rc=vec3(a,a,a);
	rc.rgb+=(fracUV.xxy-0.5)*0.15;
	return rc;
}

//#endif // ENABLE_SHADOWMAP_DEMO_EXTRAPOLATE

#ifdef ENABLE_SHADOWMAP_DEMO_BORDERSEARCH

void GetBilinearlyFilteredSample(out vec3 interpVal_DX_DY,
	vec2 baseUV, vec2 fracUV, vec2 singleUVStep)
{
	vec4 colAA=texture2D(texEdgeHint, baseUV + (vec2(0,0)*singleUVStep) );
	vec4 colBA=texture2D(texEdgeHint, baseUV + (vec2(1,0)*singleUVStep) );
	vec4 colAB=texture2D(texEdgeHint, baseUV + (vec2(0,1)*singleUVStep) );
	vec4 colBB=texture2D(texEdgeHint, baseUV + (vec2(1,1)*singleUVStep) );
	
	vec4 rc=
		(colAA*(1.0-fracUV.x)*(1.0-fracUV.y))+
		(colBA*(0.0+fracUV.x)*(1.0-fracUV.y))+
		(colAB*(1.0-fracUV.x)*(0.0+fracUV.y))+
		(colBB*(0.0+fracUV.x)*(0.0+fracUV.y));
	vec4 avXA=(colAA+colAB)*0.5;
	vec4 avXB=(colBA+colBB)*0.5;
	vec4 avYA=(colAA+colBA)*0.5;
	vec4 avYB=(colAB+colBB)*0.5;

	vec4 derivX=(avXB-avXA)*0.5;
	vec4 derivY=(avYB-avYA)*0.5;
	
	interpVal_DX_DY.x=rc.r;
	interpVal_DX_DY.y=derivX.r;
	interpVal_DX_DY.z=derivY.r;
}

vec3 GetShadowMapSample_BorderSearch(vec2 uv)
{
	vec3 rc=0.0;
	vec2 expUV=uv*float(maxRenderSize);
	vec2 fracUV=frac(expUV);
	vec2 baseUV=((expUV-fracUV)+0.5)/float(maxRenderSize);
	vec2 singleUVStep=1.0/float(maxRenderSize);
	
	vec3 val;	GetBilinearlyFilteredSample(val, baseUV, fracUV, singleUVStep);
	vec3 valNX, valPX, valNY, valPY;
	GetBilinearlyFilteredSample(valNX, baseUV+(singleUVStep*vec2(-1, 0)), fracUV, singleUVStep);
	GetBilinearlyFilteredSample(valPX, baseUV+(singleUVStep*vec2(+1, 0)), fracUV, singleUVStep);
	GetBilinearlyFilteredSample(valNY, baseUV+(singleUVStep*vec2( 0,-1)), fracUV, singleUVStep);
	GetBilinearlyFilteredSample(valPY, baseUV+(singleUVStep*vec2( 0,+1)), fracUV, singleUVStep);

	vec3 valNN, valPN, valNP, valPP;
	GetBilinearlyFilteredSample(valNN, baseUV+(singleUVStep*vec2(-1,-1)), fracUV, singleUVStep);
	GetBilinearlyFilteredSample(valPN, baseUV+(singleUVStep*vec2(+1,-1)), fracUV, singleUVStep);
	GetBilinearlyFilteredSample(valNP, baseUV+(singleUVStep*vec2(-1,+1)), fracUV, singleUVStep);
	GetBilinearlyFilteredSample(valPP, baseUV+(singleUVStep*vec2(+1,+1)), fracUV, singleUVStep);

	float interpDepth=1.0;
	vec2 samplePt=vec2(0.5, 0.5);
	
	float s=0.11;
	if(0)
	{
		if((valNX.r>s) && (valNX.r<1.0-s) && (-valNX.g<0.0))	{	samplePt=saturate(0.5 + (sign(valNX.yz)*sign(valNX.r-0.25)));	}
		if((valPX.r>s) && (valPX.r<1.0-s) && (-valNX.g>0.0))	{	samplePt=saturate(0.5 + (sign(valPX.yz)*sign(valPX.r-0.25)));	}
		if((valNY.r>s) && (valNY.r<1.0-s) && (-valNX.b<0.0))	{	samplePt=saturate(0.5 + (sign(valNY.yz)*sign(valNY.r-0.25)));	}
		if((valPY.r>s) && (valPY.r<1.0-s) && (-valNX.b>0.0))	{	samplePt=saturate(0.5 + (sign(valPY.yz)*sign(valPY.r-0.25)));	}
	}
	if(1)
	{
		  if((valNN.r>s) && (valNN.r<1.0-s) && (dot(valNN.gb, vec2(-1.0, -1.0))>0.0))	{	samplePt=saturate(vec2(0.5, 0.5) + vec2(-0.5, -0.5)*sign(valNN.r-0.5));	}
		  if((valPN.r>s) && (valPN.r<1.0-s) && (dot(valPN.gb, vec2(+1.0, -1.0))>0.0))	{	samplePt=saturate(vec2(0.5, 0.5) + vec2(+0.5, -0.5)*sign(valPN.r-0.5));	}
		  if((valNP.r>s) && (valNP.r<1.0-s) && (dot(valNP.gb, vec2(-1.0, +1.0))>0.0))	{	samplePt=saturate(vec2(0.5, 0.5) + vec2(-0.5, +0.5)*sign(valNP.r-0.5));	}
		  if((valPP.r>s) && (valPP.r<1.0-s) && (dot(valPP.gb, vec2(+1.0, +1.0))>0.0))	{	samplePt=saturate(vec2(0.5, 0.5) + vec2(+0.5, +0.5)*sign(valPP.r-0.5));	}
	}

	vec4 colGuide=texture2D(texEdgeHint, baseUV + (vec2(0.4,0.4)*singleUVStep) );
	//rc.r=colGuide.a;
	
	vec2 texOffset=(vec2(2.0, 2.0)*(samplePt+vec2(-0.5, +0.5))) + vec2(-1.5, 1.5);
	vec2 uvInterp=baseUV + (texOffset*singleUVStep);
	vec4 colInterp=texture2D(texEdgeHint,  uvInterp);
	rc.r=colInterp.a;
rc.rgb=samplePt.xxy;	
rc.rgb=colInterp.aaa;
//rc.rgb=uvInterp.xxy*20;
//rc.r=lerp(rc.r, colGuide.a, 0.4);
	return rc;
}

#endif // ENABLE_SHADOWMAP_DEMO_BORDERSEARCH

///////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_UPRES_DEMO_EXTRAPOLATE

vec3 GetUpresExample_Extrapolate(vec2 uv)
{
float upscaleFactor=2.0;	//2.0;
uv=uv/upscaleFactor;	// DEVHACK -  Blow up pixels for visibility
//uv=uv*1280.0/1920.0;
	vec3 rc;
	vec2 expUV=uv*float(maxRenderSize);
	vec2 fracUV=frac(expUV);
	vec2 baseUV=((expUV-fracUV)+0.5)/float(maxRenderSize);
	vec2 singleUVStep=1.0/float(maxRenderSize);
	// fracUV is expected to be 0.25 or 0.75
	if(0)
	{
		fracUV.x=(fracUV.x>=0.5) ? 0.75 : 0.25;
		fracUV.y=(fracUV.y>=0.5) ? 0.75 : 0.25;
	}
	
	// baseUV, fracUV are now calculated
	vec4 edgeHintAA=texture2D(texEdgeHint, baseUV + (vec2(0,0)*singleUVStep) );
	vec4 edgeHintBA=texture2D(texEdgeHint, baseUV + (vec2(1,0)*singleUVStep) );
	vec4 edgeHintAB=texture2D(texEdgeHint, baseUV + (vec2(0,1)*singleUVStep) );
	vec4 edgeHintBB=texture2D(texEdgeHint, baseUV + (vec2(1,1)*singleUVStep) );

	vec4 colAA=texture2D(texColor, baseUV + (vec2(0,0)*singleUVStep) );
	vec4 colBA=texture2D(texColor, baseUV + (vec2(1,0)*singleUVStep) );
	vec4 colAB=texture2D(texColor, baseUV + (vec2(0,1)*singleUVStep) );
	vec4 colBB=texture2D(texColor, baseUV + (vec2(1,1)*singleUVStep) );
	
	vec3 weightFuncAA=CalcWeightFunction(vec2(0,0), edgeHintAA, 0.02);
	vec3 weightFuncBA=CalcWeightFunction(vec2(1,0), edgeHintBA, 0.02);
	vec3 weightFuncAB=CalcWeightFunction(vec2(0,1), edgeHintAB, 0.02);
	vec3 weightFuncBB=CalcWeightFunction(vec2(1,1), edgeHintBB, 0.02); 
		
	vec3 fracPosH=vec3(fracUV.x, fracUV.y, 1.0);
	float weightAA=dot(weightFuncAA, fracPosH);
	float weightBA=dot(weightFuncBA, fracPosH);
	float weightAB=dot(weightFuncAB, fracPosH);
	float weightBB=dot(weightFuncBB, fracPosH);

	// Second method for calculating coverage+func: max
	float minW=10.0;
	float minWFloor=-0.5;
	vec3 minWeightFunc=vec3(0.5, 0.5, 1.0);
	vec4 minKeyCornerColor=0;
	vec4 minOppCornerColor=0;
	if((weightAA>minWFloor) && (weightAA<minW))	{	minW=weightAA;	minWeightFunc=weightFuncAA;	minKeyCornerColor=colAA;	minOppCornerColor=colBB;	}
	if((weightBA>minWFloor) && (weightBA<minW))	{	minW=weightBA;	minWeightFunc=weightFuncBA;	minKeyCornerColor=colBA;	minOppCornerColor=colAB;	}
	if((weightAB>minWFloor) && (weightAB<minW))	{	minW=weightAB;	minWeightFunc=weightFuncAB;	minKeyCornerColor=colAB;	minOppCornerColor=colBA;	}
	if((weightBB>minWFloor) && (weightBB<minW))	{	minW=weightBB;	minWeightFunc=weightFuncBB;	minKeyCornerColor=colBB;	minOppCornerColor=colAA;	}
	
	float maxW=-10.0;
	vec3 maxWeightFunc=vec3(0.5, 0.5, 1.0);
	vec4 maxKeyCornerColor=0;
	vec4 maxOppCornerColor=0;
	if(weightAA>maxW)	{	maxW=weightAA;	maxWeightFunc=weightFuncAA;	maxKeyCornerColor=colAA;	maxOppCornerColor=colBB;	}
	if(weightBA>maxW)	{	maxW=weightBA;	maxWeightFunc=weightFuncBA;	maxKeyCornerColor=colBA;	maxOppCornerColor=colAB;	}
	if(weightAB>maxW)	{	maxW=weightAB;	maxWeightFunc=weightFuncAB;	maxKeyCornerColor=colAB;	maxOppCornerColor=colBA;	}
	if(weightBB>maxW)	{	maxW=weightBB;	maxWeightFunc=weightFuncBB;	maxKeyCornerColor=colBB;	maxOppCornerColor=colAA;	}

	// Basic case: bilinear filtering
	rc.rgb=lerp(
		lerp(colAA.rgb, colBA.rgb, fracUV.x),
		lerp(colAB.rgb, colBB.rgb, fracUV.x),
		fracUV.y);	
	bool isTransitionQuad=true;
	if(	(edgeHintAA.b==1) && (edgeHintAB.b==1) &&
		(edgeHintBA.b==1) && (edgeHintBB.b==1))
	{	// Could check sum of the four b terms against 4 instead.
		isTransitionQuad=false;
	}
	if(isTransitionQuad)
	{
		rc.rgb=lerp(minKeyCornerColor.rgb, maxKeyCornerColor.rgb, saturate((0.0-(minW*1.0*upscaleFactor))));
	}
//rc=edgeHintAB;
	return rc;
	
#if 0
	float coverage=saturate(w*2);		// *2 accounts for the resolution doubling
	// Find neighbours and blur accordingly
	vec2 nebStep=vec2(sign(-weightFunc.x), sign(-weightFunc.y));
	// Fetch colors of the two relevant points
	vec3 colA=texture2D(texColor, baseUV + (fracUV*singleUVStep) );
	vec3 colB=texture2D(texColor, baseUV + (saturate(fracUV+nebStep*0.5)*singleUVStep) );
	if(1)
	{	
		rc=cornerColor.rgb;
		if(	(abs(weightFunc.r)>0.05) &&
			(abs(weightFunc.g)>0.05))
		{
			rc.rgb=lerp(colB.rgb, colA.rgb, coverage);
			rc.rgb=cornerColor;
		}
		else if(
			(edgeHintAA.z==edgeHintAB.z) &&
			(edgeHintAA.z==edgeHintBA.z) &&
			(edgeHintAA.z==edgeHintBB.z))
		{
			rc.rgb=lerp(
				lerp(colAA.rgb, colBA.rgb, fracUV.x),
				lerp(colAB.rgb, colBB.rgb, fracUV.x),
				fracUV.y);	
		}
	}
//rc.rgb=saturate(0.5 + nebStep.xxy*0.2)*coverage*2;
	//rc.rgb+=(fracUV.xxy-0.5)*0.15;
	//rc.rgb=coverage;
	//rc.rgb=weightFunc.rgb*coverage;
	
	//rc.rgb=fracUV.xxy;
#endif // 0
	return rc;
}

#endif // ENABLE_UPRES_DEMO_EXTRAPOLATE

void main(void)
{
	vec4 rc=vec4(0.2, 0.6, 0.5, 1.0);

	vec4 colRoot=texture2D(texColor, texCoord );
	vec4 colNN=tex2DOffset( texCoord, vec2(-1,-1) );
	vec4 colNP=tex2DOffset( texCoord, vec2(-1,+1) );
	vec4 colPN=tex2DOffset( texCoord, vec2(+1,-1) );
	vec4 colPP=tex2DOffset( texCoord, vec2(+1,+1) );

	// Unpack the encoded offset and coverage
	float encodedAAVal=colRoot.a*255.0;
	vec3 unpackedAAVal=frac(	vec3(encodedAAVal/256.0, encodedAAVal/128.0, encodedAAVal/64.0) );
	int sampleIndex=0;
	if(unpackedAAVal.x>=0.5)	sampleIndex+=2;
	if(unpackedAAVal.y>=0.5)	sampleIndex+=1;
	vec4 colNeb=vec4(0.0, 0.0, 0.0, 0.0);
	if(sampleIndex==0)	colNeb=colNN;
	if(sampleIndex==1)	colNeb=colNP;
	if(sampleIndex==2)	colNeb=colPN;
	if(sampleIndex==3)	colNeb=colPP;
	float coverage=unpackedAAVal.z;
	rc.rgb=lerp(colNeb.rgb, colRoot.rgb, coverage);

#ifdef ENABLE_SHADOWMAP_DEMO_EXTRAPOLATE
	rc.rgb=GetShadowMapSample_Extrapolate(texCoord/18.50);
#endif // ENABLE_SHADOWMAP_DEMO

#ifdef ENABLE_SHADOWMAP_DEMO_BORDERSEARCH
	rc.rgb=GetShadowMapSample_BorderSearch(texCoord/18.50);
#endif // ENABLE_SHADOWMAP_DEMO_BORDERSEARCH

#ifdef ENABLE_UPRES_DEMO_EXTRAPOLATE
	rc.rgb=GetUpresExample_Extrapolate(texCoord);
#endif // ENABLE_UPRES_DEMO_EXTRAPOLATE


//rc.rgb=colNeb.rgb;
//rc.rgb=float(sampleIndex)/3.0;	// DEVHACK
//rc.rgb*=0.7;
//rc.rgb=colRoot.rgb;
//rc.rgb=texCoord.xxy;
//rc.rgb=texture2D(texEdgeHint, texCoord*0.1).rgb;
//rc.rgb=texture2D(texColor, texCoord*0.1).rgb;
if(0)
{
	outColor.r=unpackedAAVal.z;
	outColor.g=(128.0+sign(unpackedAAVal.x)*64.0)/255.0;
	outColor.b=(128.0+sign(unpackedAAVal.y)*64.0)/255.0;
}

//if(1)	rc.rgb=colRoot.rgb;
//rc.rgb=coverage;
rc.a=1.0;
	gl_FragColor=rc;
}
