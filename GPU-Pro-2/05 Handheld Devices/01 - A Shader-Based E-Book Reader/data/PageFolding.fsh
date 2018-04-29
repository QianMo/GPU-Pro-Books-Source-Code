// All Defines:
//#define HAS_TEXTURE
//#define EDGE_GRADIENT
//#define EDGE_GRADIENT_INT
//#define EDGE_GRADIENT_EXT
//#define FOLD_GRADIENT
//#define ENABLE_WIREFRAME

#ifdef HAS_TEXTURE
uniform sampler2D  	sTexture;
varying mediump vec2   TexCoord;
#endif

#ifdef EDGE_GRADIENT
varying lowp float EdgeGradient;
#endif

#ifdef FOLD_GRADIENT
varying lowp float FoldGradient;
#endif

void main()
{
#ifdef EDGE_GRADIENT_EXT
	gl_FragColor = vec4(0.5, 0.5, 0.5, EdgeGradient);
#else
	#ifdef HAS_TEXTURE
	#ifdef EDGE_GRADIENT
		gl_FragColor = texture2D(sTexture, TexCoord) - EdgeGradient;
	#else
		#ifndef ENABLE_WIREFRAME
			gl_FragColor = texture2D(sTexture, TexCoord);
		#else
			gl_FragColor = vec4(0.0);
		#endif
	#endif
	#endif

	#ifdef FOLD_GRADIENT
		gl_FragColor = mix(gl_FragColor, vec4(0.5), FoldGradient);
	#endif
#endif
}