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

#ifndef __MATRIX4__H_
#define __MATRIX4__H_

#include "Geometry.h"

struct Matrix4
{
	union {
		struct {
			float xx,xy,xz,xt;
			float yx,yy,yz,yt;
			float zx,zy,zz,zt;
			float tx,ty,tz,tt;
		};
		float m[4][4];
		float _m[16];
	};

	Matrix4()
	{
		xx = xy = xz = xt = 0;
		yx = yy = yz = yt = 0;
		zx = zy = zz = zt = 0;
		tx = ty = tz = tt = 0;
	}
};

inline void M4_MakeOrthoNormal(Matrix4 &T)
{
	Vector3 auxvx(T.xx,T.xy,T.xz);
	Vector3 auxvy(T.yx,T.yy,T.yz);
	Vector3 auxvz(T.zx,T.zy,T.zz);
	Normalize(auxvx);
	float prod;
	Dot(prod, auxvx,auxvy);
	auxvy=auxvy-prod*auxvx;
	Normalize(auxvy);
	Dot(prod,auxvx,auxvz);
	auxvz=auxvz-prod*auxvx;
	Dot(prod,auxvy,auxvz);
	auxvz=auxvz-prod*auxvy;
	Normalize(auxvy);
	
	T.xx = auxvx.x;T.xy = auxvx.y;T.xz = auxvx.z;
	T.yx = auxvy.x;T.yy = auxvy.y;T.yz = auxvy.z;
	T.zx = auxvz.x;T.zy = auxvz.y;T.zz = auxvz.z;
}

inline void M4_Transform(Vector3 &Out,const Vector3 &In,const Matrix4 &T)
{
	Out.x = T.xx*In.x+T.xy*In.y+T.xz*In.z+T.xt;
	Out.y = T.yx*In.x+T.yy*In.y+T.yz*In.z+T.yt;
	Out.z = T.zx*In.x+T.zy*In.y+T.zz*In.z+T.zt;
}

inline void M4_TransformOnlyMatrix(Vector3 &Out,const Vector3 &In,const Matrix4 &T)
{
	Out.x = T.xx*In.x+T.xy*In.y+T.xz*In.z;
	Out.y = T.yx*In.x+T.yy*In.y+T.yz*In.z;
	Out.z = T.zx*In.x+T.zy*In.y+T.zz*In.z;
}

inline void M4_Identity(Matrix4 &T)
{
	T.xx=T.yy=T.zz=T.tt=1.0f;
	T.xy=T.xz=T.xt=
	T.yx=T.yz=T.yt=
	T.zx=T.zy=T.zt=
	T.tx=T.ty=T.tz=0.0f;
}

inline void M4_Translation(Matrix4 &T,float x,float y,float z)
{
	T.xx=T.yy=T.zz=T.tt=1.0f;
	T.xy=T.xz=
	T.yx=T.yz=
	T.zx=T.zy=
	T.tx=T.ty=T.tz=0.0f;
	T.xt=x;
	T.yt=y;
	T.zt=z;
}

inline void M4_Rotation(Matrix4 &T,float x,float y,float z,float angle)
{
	float norm = sqrt(x*x+y*y+z*z);
	if (norm>0.0f)
	{
		x/=norm;
		y/=norm;
		z/=norm;
	}
	
	float c = cos(angle);
	float cminus = 1.0f-c;
	float s = sin(angle);
	Vector3 auxv(x,y,z);
	Vector3 auxvx(c,-z*s,y*s);
	auxvx = auxvx + (cminus*x)*auxv;
	Vector3 auxvy(z*s,c,-x*s);
	auxvy = auxvy + (cminus*y)*auxv;
	Vector3 auxvz(-y*s,x*s,c);
	auxvz = auxvz + (cminus*z)*auxv;
	// make sure the matriz is orthonormal
	Normalize(auxvx);
	float prod;
	Dot(prod, auxvx,auxvy);
	auxvy=auxvy-prod*auxvx;
	Normalize(auxvy);
	Dot(prod, auxvx,auxvz);
	auxvz=auxvz-prod*auxvx;
	Dot(prod,auxvy,auxvz);
	auxvz=auxvz-prod*auxvy;
	Normalize(auxvy);
	
	T.xx = auxvx.x;T.xy = auxvx.y;T.xz = auxvx.z;T.xt=0.0f;
	T.yx = auxvy.x;T.yy = auxvy.y;T.yz = auxvy.z;T.yt=0.0f;
	T.zx = auxvz.x;T.zy = auxvz.y;T.zz = auxvz.z;T.zt=0.0f;
	T.tx = T.ty = T.tz = 0.0f;
	T.tt = 1.0f;
}

inline void M4_Compose(Matrix4 &Out,const Matrix4 &First,const Matrix4 &Second)
{
	// makes Out = Second(First()); == M2*M1+M2*T1+T2
	Out.xx = Second.xx*First.xx + Second.xy*First.yx + Second.xz*First.zx;
	Out.xy = Second.xx*First.xy + Second.xy*First.yy + Second.xz*First.zy;
	Out.xz = Second.xx*First.xz + Second.xy*First.yz + Second.xz*First.zz;
	Out.xt = Second.xx*First.xt + Second.xy*First.yt + Second.xz*First.zt + Second.xt;

	Out.yx = Second.yx*First.xx + Second.yy*First.yx + Second.yz*First.zx;
	Out.yy = Second.yx*First.xy + Second.yy*First.yy + Second.yz*First.zy;
	Out.yz = Second.yx*First.xz + Second.yy*First.yz + Second.yz*First.zz;
	Out.yt = Second.yx*First.xt + Second.yy*First.yt + Second.yz*First.zt + Second.yt;

	Out.zx = Second.zx*First.xx + Second.zy*First.yx + Second.zz*First.zx;
	Out.zy = Second.zx*First.xy + Second.zy*First.yy + Second.zz*First.zy;
	Out.zz = Second.zx*First.xz + Second.zy*First.yz + Second.zz*First.zz;
	Out.zt = Second.zx*First.xt + Second.zy*First.yt + Second.zz*First.zt + Second.zt;

	Out.tx = 0.0f;
	Out.ty = 0.0f;
	Out.tz = 0.0f;
	Out.tt = 1.0f;
}

inline void M4_Inverse(Matrix4 &Out,const Matrix4 &In)
{
	Out.xx = In.xx; Out.xy = In.yx; Out.xz = In.zx;
	Out.yx = In.xy; Out.yy = In.yy; Out.yz = In.zy;
	Out.zx = In.xz; Out.zy = In.yz; Out.zz = In.zz;
	Vector3 Aux(-In.xt,-In.yt,-In.zt);
	Vector3 AuxRes(0.f,0.f,0.f);
	M4_Transform(AuxRes,Aux,Out);
	Out.xt=AuxRes.x;
	Out.yt=AuxRes.y;
	Out.zt=AuxRes.z;
}

inline void M4_Transpose(Matrix4 &Out,const Matrix4 &In)
{
	Out.xx = In.xx; Out.xy = In.yx; Out.xz = In.zx; Out.xt = In.tx;
	Out.yx = In.xy; Out.yy = In.yy; Out.yz = In.zy; Out.yt = In.ty;
	Out.zx = In.xz; Out.zy = In.yz; Out.zz = In.zz; Out.zt = In.tz;
	Out.tx = In.xt; Out.ty = In.yt; Out.tz = In.zt; Out.tt = In.tt;
}

#endif