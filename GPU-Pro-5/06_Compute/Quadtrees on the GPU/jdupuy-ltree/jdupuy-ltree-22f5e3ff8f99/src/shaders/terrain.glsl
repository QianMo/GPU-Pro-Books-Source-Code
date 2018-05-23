#ifndef TERRAIN_GLSL
#define TERRAIN_GLSL

#ifndef PI
#	define PI 3.14159265
#endif

// noise sampler
uniform sampler2D u_noise_sampler;
uniform sampler2D u_puget_sampler;

const float lacunarity = 2.0;
const float max_octaves = 4.0;
const float H = 0.96;
const float p_scale = 0.03125 * 0.01; // first factor is original spacing for noise (256/8)
const float amp = 5.0;

float
displace(vec2 p) {
	float octaves = max_octaves;
	float value = 0.0;
	p*= p_scale;
	float f = 1.5;

	for(float i=0.0; i<octaves; i+=1.0) {
		float z = texture(u_noise_sampler, p).r;//*2.0-1.0;
		value+= z * pow(f, -H);
		f*= lacunarity;
		p*= lacunarity;
	}
	value+= fract(octaves) * texture(u_noise_sampler, p).r * pow(f, -H);
	return value*amp;
}

float
displace(vec2 p, float screen_resolution) {
	return texture(u_puget_sampler, p).r * 6553.5;
#if 0
	float octaves = clamp(log2(screen_resolution) - 2.0, 0.0, max_octaves);
	float value = 0.0;
	p*= p_scale;
	float f = 1.5;

	for(float i=0.0; i<octaves; i+=1.0) {
		float z = texture(u_noise_sampler, p).r;//*2.0-1.0;
		value+= z * pow(f, -H);
		f*= lacunarity;
		p*= lacunarity;
	}
	value+= fract(octaves) * texture(u_noise_sampler, p).r * pow(f, -H);
	return value*amp;
#endif
}

#endif //TERRAIN_GLSL

