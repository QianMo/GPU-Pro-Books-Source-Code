/*
Copyright (c) 2016 SIDIA

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef SHADER_UTILITIES
#define SHADER_UTILITIES

#define LINEAR_SPACE_RT_SUPPORTED 0

#define DEGREE_TO_RAD 0.01745329251994
#define BRS_SUPPORTED_TARGETS opengl gles gles3 dx9 

float Deg2Rad(float angleDegrees)
{
	return DEGREE_TO_RAD * angleDegrees;
}

float2 NormalizeScreenSpace(float4 screenPos)
{
	//////////////////////////////////////////////////////////////////////////////////////
    // Screen UV Mapping - We get a normalized tiled coordinate with (0, 0) being at the /
    // bottom left corner of the screen                                                  /  
    //////////////////////////////////////////////////////////////////////////////////////
    float2 screenUV = screenPos.xy / screenPos.w;
    float aspectRatio = _ScreenParams.x / _ScreenParams.y;
    
    // apply aspect ratio and convert from (0, 0) at center to (0, 0) at bottom left.
    screenUV = float2(screenUV.x + 1.0, screenUV.y + 1.0) * 0.5;
    screenUV.x *= aspectRatio;
    screenUV.y *= _ProjectionParams.x;
    return screenUV;
}

float2 RotateUV(float2 uv, float angle)
{                
	float angleRad = Deg2Rad(angle);
    float cosAngle = cos(angleRad);
    float sinAngle = sin(angleRad);
                
    float2x2 rotMatrix = float2x2(cosAngle, -sinAngle, sinAngle, cosAngle);
    float2 rotatedScreenUV = mul(rotMatrix, uv);
    
    return rotatedScreenUV;
}

half HalfTone(half RepeatRate , half DotSize , half2 UV)
{
	half size = 1.0 / RepeatRate;
	half2 cellSize = half2(size, size);
    half2 cellCenter = cellSize * 0.5;
    
    // UV is normalized into range [0, 1] but since it gets a rotation matrix
    // applied, values can be now negatives. Thats why we use abs.
    half2 uvlocal = fmod(abs(UV), cellSize);
    half distance = length(uvlocal - cellCenter);
    half radius = (cellSize.x * 0.5) * DotSize;
    
	// Anti-Aliasing based on differentials
	half threshold = length(ddx(distance) - ddy(distance));
	return smoothstep(distance - threshold, distance + threshold, radius);
}

half Softlight(half b, half a)
{
	if (b < 0.5)
	{
		// 2ab - a² (1 - 2b)
		return 2.0 * a * b + a * a * (1.0 - 2.0 * b);
	}
	else
	{
		// 2a (1 - b) + sqrt(a) (2b - 1)
		return 2.0 * a * (1.0 - b) + sqrt(a) * (2.0 * b - 1.0);
	}
}

float4 tex2dLinear(sampler2D tex, float2 uv)
{
#if LINEAR_SPACE_RT_SUPPORTED
	return tex2D(tex, uv);
#else
	return pow(tex2D(tex, uv), 2.2);
#endif
}

fixed4 gammaConvert(float4 color)
{
#if LINEAR_SPACE_RT_SUPPORTED
	return color;
#else
	return pow(color, 0.454545);
#endif
}
#endif