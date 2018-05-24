#ifndef SHADERCOMMON
#define SHADERCOMMON

struct LinkedLightID
{
	uint lightID;
	uint link;
};

float LinearDepth(float nonLinear)
{
	float c1 = FARZ / NEARZ;
	float c0 = 1.0 - c1;
	return FARZ / (c0 * nonLinear + c1);
	//(FARZ * NEARZ)/(FARZ * (-nonLinear) + FARZ + NEARZ * nonLinear)
}

float LinearDepthOne(float nonLinear)
{
	float c1 = FARZ / NEARZ;
	float c0 = 1.0 - c1;
	return 1.0f / (c0 * nonLinear + c1);
	//(FARZ * NEARZ)/(FARZ * (-nonLinear) + FARZ + NEARZ * nonLinear)
}

#endif
