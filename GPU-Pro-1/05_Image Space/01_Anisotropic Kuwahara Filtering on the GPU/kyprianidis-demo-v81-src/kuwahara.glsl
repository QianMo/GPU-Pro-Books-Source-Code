// Anisotropic Kuwahara Filtering on the GPU
// by Jan Eric Kyprianidis <www.kyprianidis.com>
#version 120
#extension GL_EXT_gpu_shader4 : enable

uniform sampler2D src;
uniform int radius;

void main (void) {
	vec2 src_size = vec2(textureSize2D(src, 0));
	vec2 uv = gl_FragCoord.xy / src_size;
	float n = float((radius + 1) * (radius + 1));

	vec3 m[4];
	vec3 s[4];
	for (int k = 0; k < 4; ++k) {
		m[k] = vec3(0.0);
		s[k] = vec3(0.0);
	}

	for (int j = -radius; j <= 0; ++j)  {
		for (int i = -radius; i <= 0; ++i)  {
			vec3 c = texture2D(src, uv + vec2(i,j) / src_size).rgb;
			m[0] += c;
			s[0] += c * c;
		}
	}

	for (int j = -radius; j <= 0; ++j)  {
		for (int i = 0; i <= radius; ++i)  {
			vec3 c = texture2D(src, uv + vec2(i,j) / src_size).rgb;
			m[1] += c;
			s[1] += c * c;
		}
	}

	for (int j = 0; j <= radius; ++j)  {
		for (int i = 0; i <= radius; ++i)  {
			vec3 c = texture2D(src, uv + vec2(i,j) / src_size).rgb;
			m[2] += c;
			s[2] += c * c;
		}
	}

	for (int j = 0; j <= radius; ++j)  {
		for (int i = -radius; i <= 0; ++i)  {
			vec3 c = texture2D(src, uv + vec2(i,j) / src_size).rgb;
			m[3] += c;
			s[3] += c * c;
		}
	}


	float min_sigma2 = 1e+2;
	for (int k = 0; k < 4; ++k) {
		m[k] /= n;
		s[k] = abs(s[k] / n - m[k] * m[k]);

		float sigma2 = s[k].r + s[k].g + s[k].b;
		if (sigma2 < min_sigma2) {
			min_sigma2 = sigma2;
			gl_FragColor = vec4(m[k], 1.0);
		}
	}
}
