uniform sampler2D tex;

uniform float gammaExponent;

void main()
{
   gl_FragColor = pow(texture2D(tex, gl_TexCoord[0].st), vec4(gammaExponent));
   gl_FragColor.a = 1.0;

}

