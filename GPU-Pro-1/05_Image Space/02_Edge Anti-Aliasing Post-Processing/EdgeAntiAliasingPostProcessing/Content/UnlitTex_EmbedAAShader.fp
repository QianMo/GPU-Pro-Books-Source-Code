varying vec2 texCoord;
uniform sampler2D textureMap;

vec3 frac(vec3 x)	{	return x - floor(x);	}

void main(void)
{
	vec4 outColor=0;
	vec4 outEdgeHint=0;
	
	outColor=texture2D(textureMap, texCoord);
//gl_FragColor.rgb=vec3(0.2, 0.8, 0.5);	// DEVHACK

#ifdef ENABLE_SHADOWMAP_DEMO_EXTRAPOLATE
	outEdgeHint.rgb=vec3(0.5, 0.5, 0.0);	// DEVHACK
#endif // ENABLE_SHADOWMAP_DEMO_EXTRAPOLATE

#ifdef ENABLE_SHADOWMAP_DEMO_BORDERSEARCH
	outEdgeHint.rgb=vec3(0.0, 0.0, 0.0);	// DEVHACK
#endif // ENABLE_SHADOWMAP_DEMO_BORDERSEARCH

#ifdef ENABLE_UPRES_DEMO_EXTRAPOLATE
	outEdgeHint.rgb=vec3(0.5, 0.5, 1.0);	// DEVHACK
#endif // ENABLE_UPRES_DEMO_EXTRAPOLATE

#ifdef ENABLE_ALLEDGEBLUR_DEMO
	outEdgeHint.rgb=vec3(0.5, 0.5, 1.0);	// DEVHACK
#endif // ENABLE_ALLEDGEBLUR_DEMO

	gl_FragData[0]=outColor;
	gl_FragData[1]=outEdgeHint;
}
