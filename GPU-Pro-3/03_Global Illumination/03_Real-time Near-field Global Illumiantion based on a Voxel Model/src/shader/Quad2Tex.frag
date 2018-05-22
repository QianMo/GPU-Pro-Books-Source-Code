uniform sampler2D tex0;
uniform sampler2D tex1;

uniform bool tex1SingleChannel;
uniform int operation; // 0 = Multiply, 1 = Add
//uniform float factor;

void main()
{
   vec4 val0 = texture2D(tex0, gl_TexCoord[0].st);
   vec4 val1 = vec4(1);
   if(tex1SingleChannel)
   {
      val1 = vec4(texture2D(tex1, gl_TexCoord[0].st).r);
   }
   else
   {
      val1 = texture2D(tex1, gl_TexCoord[0].st);
   }
   if(operation == 0)
   {
      gl_FragColor = val0 * val1;// * factor;
   }
   else if(operation == 1)
   {
      gl_FragColor = val0 + val1;
   }
}