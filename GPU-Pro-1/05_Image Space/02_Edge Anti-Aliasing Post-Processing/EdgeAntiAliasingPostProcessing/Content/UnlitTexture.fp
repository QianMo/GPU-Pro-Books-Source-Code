varying vec2 texCoord;
uniform sampler2D textureMap;

vec3 frac(vec3 x)	{	return x - floor(x);	}

void main(void)
{
	gl_FragColor=texture2D(textureMap, texCoord);
//gl_FragColor.rgb=vec3(0.2, 0.8, 0.5);	// DEVHACK

#ifdef ENABLE_SHADOWMAP_DEMO_EXTRAPOLATE
	gl_FragColor.rgb=vec3(0.5, 0.5, 0.0);	// DEVHACK
#endif // ENABLE_SHADOWMAP_DEMO_EXTRAPOLATE

#ifdef ENABLE_SHADOWMAP_DEMO_BORDERSEARCH
	gl_FragColor.rgb=vec3(0.0, 0.0, 0.0);	// DEVHACK
#endif // ENABLE_SHADOWMAP_DEMO_BORDERSEARCH

#ifdef ENABLE_UPRES_DEMO_EXTRAPOLATE
	gl_FragColor.rgb=vec3(0.5, 0.5, 0.0);	// DEVHACK
#endif // ENABLE_UPRES_DEMO_EXTRAPOLATE
}
