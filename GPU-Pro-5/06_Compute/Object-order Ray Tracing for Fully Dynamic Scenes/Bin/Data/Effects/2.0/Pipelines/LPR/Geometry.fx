#ifndef BE_LPR_GEOMETRY_H
#define BE_LPR_GEOMETRY_H

#include <Utility/Color.fx>

// Geometry
struct GBufferGeometry
{
	float Depth;
	float3 Normal;
};
GBufferGeometry MakeGeometry(float depth, float3 normal)
{
	GBufferGeometry o;
	o.Depth = depth;
	o.Normal = normal;
	return o;
}
GBufferGeometry ExtractGeometry(float4 geometry)
{
	return (GBufferGeometry) geometry;
}
float ExtractDepth(float4 geometry)
{
	return ExtractGeometry(geometry).Depth;
}
float3 ExtractNormal(float4 geometry)
{
	return ExtractGeometry(geometry).Normal;
}

// Diffuse
struct GBufferDiffuse
{
	float3 Color;
	float Roughness;
};
GBufferDiffuse MakeDiffuse(float3 color, float roughness)
{
	GBufferDiffuse o;
	o.Color = color;
	o.Roughness = roughness;
	return o;
}
GBufferDiffuse MakeDiffuse(float3 color)
{
	return MakeDiffuse(color, 0.0f);
}
GBufferDiffuse ExtractDiffuse(float4 diffuse)
{
	return (GBufferDiffuse) diffuse;
}

// Specular
struct GBufferSpecular
{
	float3 Color;
	float Shininess;
	float FresnelR;
	float Metalness;
	float FresnelM;
};
GBufferSpecular MakeSpecular(float3 color, float shininess, float fresnelR, float metalness, float fresnelM)
{
	GBufferSpecular o;
	o.Color = color;
	o.Shininess = shininess;
	o.FresnelR = fresnelR;
	o.Metalness = metalness;
	o.FresnelM = fresnelM;
	return o;
}
GBufferSpecular MakeSpecular(float3 color, float shininess, float fresnelR)
{
	return MakeSpecular(color, shininess, fresnelR, 0.0f, 0.0f);
}
uint PackSpecular(GBufferSpecular specular)
{
	uint o = packcolor16(sqrt(sqrt(specular.Color)));
	uint4 p = (uint4) 
		clamp(
			float4(sqrt(specular.Shininess), sqrt(specular.FresnelR), specular.Metalness, sqrt(specular.FresnelM)) * 16.0f,
			0.0f, 15.9f
		);
	p = p << uint4(16, 20, 24, 28);
	return o | p.x | p.y | p.z | p.w; // ColorShineFresMetalFres
}
GBufferSpecular UnpackSpecular(uint v)
{
	GBufferSpecular o;
	o.Color = unpackcolor16(v);
	o.Color *= o.Color;
	o.Color *= o.Color;
	
	float4 f = ( (v >> uint4(16, 20, 24, 28)) & 0xf ) / 15.0f;
	f = saturate(f);
	o.Shininess = f.x * f.x;
	o.FresnelR = f.y * f.y;
	o.Metalness = f.z;
	o.FresnelM = f.w * f.w;
	return o;
}
GBufferSpecular ExtractSpecular(uint4 specular)
{
	return UnpackSpecular(specular.x);
}

// Output binding
struct GBufferBinding
{
	float4 Geometry		: SV_Target0;
	float4 Diffuse		: SV_Target1;
	uint4 Specular		: SV_Target2;
};
GBufferBinding BindGBuffer(GBufferGeometry geometry, GBufferDiffuse diffuse, GBufferSpecular specular)
{
	GBufferBinding o;
	o.Geometry = (float4) geometry;
	o.Diffuse = (float4) diffuse;
	o.Specular = PackSpecular(specular);
	return o;
}

#endif