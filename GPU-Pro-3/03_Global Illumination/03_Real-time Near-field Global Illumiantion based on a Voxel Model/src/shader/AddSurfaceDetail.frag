uniform sampler2D positionBuffer;
uniform sampler2D normalBuffer;
uniform sampler2D inputTex;

uniform vec3 cameraPosWorldSpace;
uniform float alpha;

varying out vec3 c_out;

void main()
{
   vec3 V   = normalize(cameraPosWorldSpace - texture2D(positionBuffer, gl_TexCoord[0].xy).xyz);
   vec3 N   = texture2D(normalBuffer, gl_TexCoord[0].xy).xyz;
   vec3 c   = texture2D(inputTex, gl_TexCoord[0].xy).rgb;

   // formula according to Greg Nichols' Ph.D. thesis 
   // http://www.gregnichols.org/pubs/thesis.pdf (page 50f)
   c_out = c * (alpha * dot(V, N) + (1.0 - alpha));
}