#include <image/fourier.h>
#include <math/common.h>


using namespace NMath;


float* NImage::DiscreteFourierTransform(uint8* input, uint16 width, uint16 height)
{
	float* output = new float[2 * width * height];

	for (uint16 y = 0; y < height; y++)
	{
		for (uint16 x = 0; x < width; x++)
		{
			output[2*Idx(x, y, width) + 0] = 0.0f;
			output[2*Idx(x, y, width) + 1] = 0.0f;

			for (int j = 0; j < height; j++)
			{
				for (int i = 0; i < width; i++)
				{
					float p = -TwoPi * ((float)(x*i)/(float)width + (float)(y*j)/(float)height);

					float R = (float)input[Idx(i, j, width)];
					float I = 0.0f;

					output[2*Idx(x, y, width) + 0] += R*Cos(p) - I*Sin(p);
					output[2*Idx(x, y, width) + 1] += R*Sin(p) + I*Cos(p);
				}
			}
		}
	}

	return output;
}


float* NImage::InverseDiscreteFourierTransform(float* input, uint16 width, uint16 height)
{
	float* output = new float[2 * width * height];

	for (uint16 y = 0; y < height; y++)
	{
		for (uint16 x = 0; x < width; x++)
		{
			output[2*Idx(x, y, width) + 0] = 0.0f;
			output[2*Idx(x, y, width) + 1] = 0.0f;

			for (int j = 0; j < height; j++)
			{
				for (int i = 0; i < width; i++)
				{
					float p = TwoPi * ((float)(x*i)/(float)width + (float)(y*j)/(float)height);

					float& R = input[2*Idx(i, j, width) + 0];
					float& I = input[2*Idx(i, j, width) + 1];

					output[2*Idx(x, y, width) + 0] += R*Cos(p) - I*Sin(p);
					output[2*Idx(x, y, width) + 1] += R*Sin(p) + I*Cos(p);
				}
			}

			output[2*Idx(x, y, width) + 0] /= (float)(width * height);
			output[2*Idx(x, y, width) + 1] /= (float)(width * height);
		}
	}

	return output;
}


float* NImage::DiscreteFourierTransform_Separable(uint8* input, uint16 width, uint16 height)
{
	float* output = new float[2 * width * height];
	float* temp = new float[2 * width * height];

	for (uint16 j = 0; j < height; j++)
	{
		for (uint16 x = 0; x < width; x++)
		{
			temp[2*Idx(x, j, width) + 0] = 0.0f;
			temp[2*Idx(x, j, width) + 1] = 0.0f;

			for (int i = 0; i < width; i++)
			{
				float p = -TwoPi * ((float)(x*i)/(float)width);

				float R = (float)input[Idx(i, j, width)];
				float I = 0.0f;

				temp[2*Idx(x, j, width) + 0] += R*Cos(p) - I*Sin(p);
				temp[2*Idx(x, j, width) + 1] += R*Sin(p) + I*Cos(p);
			}
		}
	}

	for (uint16 y = 0; y < height; y++)
	{
		for (uint16 x = 0; x < width; x++)
		{
			output[2*Idx(x, y, width) + 0] = 0.0f;
			output[2*Idx(x, y, width) + 1] = 0.0f;

			for (int j = 0; j < height; j++)
			{
				float p = -TwoPi * ((float)(y*j)/(float)height);

				float& R = temp[2*Idx(x, j, width) + 0];
				float& I = temp[2*Idx(x, j, width) + 1];

				output[2*Idx(x, y, width) + 0] += R*Cos(p) - I*Sin(p);
				output[2*Idx(x, y, width) + 1] += R*Sin(p) + I*Cos(p);
			}
		}
	}

	delete[] temp;
	return output;
}


float* NImage::InverseDiscreteFourierTransform_Separable(float* input, uint16 width, uint16 height)
{
	float* output = new float[2 * width * height];
	float* temp = new float[2 * width * height];

	for (uint16 j = 0; j < height; j++)
	{
		for (uint16 x = 0; x < width; x++)
		{
			temp[2*Idx(x, j, width) + 0] = 0.0f;
			temp[2*Idx(x, j, width) + 1] = 0.0f;

			for (int i = 0; i < width; i++)
			{
				float p = TwoPi * ((float)(x*i)/(float)width);

				float& R = input[2*Idx(i, j, width) + 0];
				float& I = input[2*Idx(i, j, width) + 1];

				temp[2*Idx(x, j, width) + 0] += R*Cos(p) - I*Sin(p);
				temp[2*Idx(x, j, width) + 1] += R*Sin(p) + I*Cos(p);
			}

			temp[2*Idx(x, j, width) + 0] /= (float)width;
			temp[2*Idx(x, j, width) + 1] /= (float)width;
		}
	}

	for (uint16 y = 0; y < height; y++)
	{
		for (uint16 x = 0; x < width; x++)
		{
			output[2*Idx(x, y, width) + 0] = 0.0f;
			output[2*Idx(x, y, width) + 1] = 0.0f;

			for (int j = 0; j < height; j++)
			{
				float p = TwoPi * ((float)(y*j)/(float)height);

				float& R = temp[2*Idx(x, j, width) + 0];
				float& I = temp[2*Idx(x, j, width) + 1];

				output[2*Idx(x, y, width) + 0] += R*Cos(p) - I*Sin(p);
				output[2*Idx(x, y, width) + 1] += R*Sin(p) + I*Cos(p);
			}

			output[2*Idx(x, y, width) + 0] /= (float)height;
			output[2*Idx(x, y, width) + 1] /= (float)height;
		}
	}

	delete[] temp;
	return output;
}
