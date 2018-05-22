varying vec4 P;

uniform bool writeColor; // true: write color, false: write position

void main()
{
   gl_FragColor = writeColor ? gl_Color : P;
}