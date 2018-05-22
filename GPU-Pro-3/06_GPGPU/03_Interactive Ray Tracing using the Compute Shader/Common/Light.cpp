// ================================================================================ //
// Copyright (c) 2011 Arturo Garcia, Francisco Avila, Sergio Murguia and Leo Reyes	//
//																					//
// Permission is hereby granted, free of charge, to any person obtaining a copy of	//
// this software and associated documentation files (the "Software"), to deal in	//
// the Software without restriction, including without limitation the rights to		//
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies	//
// of the Software, and to permit persons to whom the Software is furnished to do	//
// so, subject to the following conditions:											//
//																					//
// The above copyright notice and this permission notice shall be included in all	//
// copies or substantial portions of the Software.									//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR		//
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,			//
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE		//
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER			//
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,	//
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE	//
// SOFTWARE.																		//
// ================================================================================ //

#include "Light.h"

Light::Light( unsigned int uiChoice )
{
	SelectLight(uiChoice);
}

Light::~Light(void)
{
}

void Light::SelectLight( unsigned int uiChoice)
{
	switch(uiChoice)
	{
	case 0:
		m_vfPosition = Vector3(1.732050807568877/4.0,2.0/4.0,-3.0/4.0);
		break;
	case 1:
		m_vfPosition = Vector3(0.f,0.2f,0.f);
		break;
	}
}

void Light::Move( float &a_fTime, float &a_fSpeed, Vector3 &a_vTraslation )
{
	float NormSQ;
	Dot(NormSQ, a_vTraslation, a_vTraslation);
	if(NormSQ > 0.1f)
	{
		// Move Light
		a_vTraslation = (-a_fTime*a_fSpeed/sqrt(NormSQ)) * a_vTraslation;
		m_vfPosition += a_vTraslation;
	}
}
