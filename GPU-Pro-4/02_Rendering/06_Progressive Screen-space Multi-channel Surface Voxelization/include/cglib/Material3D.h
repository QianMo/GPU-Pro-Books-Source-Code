#pragma once

#ifdef WIN32
	#define WIN32_LEAN_AND_MEAN
	#include <windows.h>
#else
	#include <limits.h>		// for INT_MAX
#endif

#include <GL/gl.h>

#include "Vector2D.h" 
#include "Vector3D.h" 
#include "Texture2D.h" 

enum _faceType
{
    MATERIAL_FRONT = GL_FRONT,
    MATERIAL_BACK  = GL_BACK,
    MATERIAL_FRONT_AND_BACK  = GL_FRONT_AND_BACK
};

enum _colorMode
{
    MATERIAL_AMBIENT = GL_AMBIENT,
    MATERIAL_DIFFUSE = GL_DIFFUSE,
    MATERIAL_SPECULAR = GL_SPECULAR,
    MATERIAL_EMISSION = GL_EMISSION,
    MATERIAL_AMBIENT_AND_DIFFUSE = GL_AMBIENT_AND_DIFFUSE,
    MATERIAL_OFF
};

// default is BLINN
enum _shadingType
{
    MATERIAL_NONE = 0,
    MATERIAL_BLINN,
    MATERIAL_METAL,
    MATERIAL_ROUGH
};

enum _mapType
{
    MATERIAL_MAP_DIFFUSE0 = 0,      // map_Kd   (RGBA)
    MATERIAL_MAP_DIFFUSE1,          // map_Kd2  (RGBA) 
    MATERIAL_MAP_BUMP,              // map_bump (greyscale->height)
    MATERIAL_MAP_SPECULAR,          // map_Ks   (Ks_R, Ks_G, Ks_B, Kexp)
    MATERIAL_MAP_EMISSION,          // map_Ke   (grayscale -> intensity)
    MATERIAL_MAP_COUNT
};

class Material3D
{
public: 
	Material3D();
	~Material3D();
	
	Vector3D ambient;
	Vector3D diffuse;
	Vector3D specular;
	Vector3D emission;
	int shininess;										  // for Blinn model, 
	float metallic_shine;								  // for Strauss metal model
	float roughness;									  // for Oren-Nayar approx. model
	float translucency;									  // for subsurface scattering and GI effects
	float alpha;										  // 1-transparency
	_shadingType type;									  // material (shading model) type
	float ior;											  // rel. index of refraction
	char name[MAXSTRING];								  // loaded material name
	bool has_texture[MATERIAL_MAP_COUNT];       // true if it has i-th layer of texture, follows the _mapType
	bool has_auto_normal_map;							  // true if normal map is extracted from diffuse color map 0
	char *texturestr[MATERIAL_MAP_COUNT];		  // color texture name, follows the _mapType
	unsigned int texturemap[MATERIAL_MAP_COUNT];// texture IDs, follows the _mapType
	
	void draw ();
	void dump ();
	char * dumpType ();
	
	static void generateAutoNormalMap (Texture2D &normal_map, Texture2D color_map);
	static void generateNormalMapFromBumpMap (Texture2D &normal_map, Texture2D bump_map);
	
	inline void  setAmbient (const Vector3D & a) { ambient = a; }
	inline const Vector3D & getAmbient() const { return ambient; }
	
	inline void  setDiffuse (const Vector3D & d) { diffuse = d; }
	inline const Vector3D & getDiffuse() const { return diffuse; }
	
	inline void  setSpecular (const Vector3D & s) { specular = s; }
	inline const Vector3D & getSpecular() const { return specular; }
	
	inline void  setEmission (const Vector3D & s) { emission = s; }
	inline const Vector3D & getEmission() const { return emission; }
	
	inline void  setShininess (int s) { shininess = s; }
	inline const int getShininess () const { return shininess; }
	
	inline void  setMetallicShine (float s) { metallic_shine = s; }
	inline const float getMetallicShine () const { return metallic_shine; }
	
	inline void  setRoughness (float r) { roughness = r; }
	inline const float getRoughness () const { return roughness; }
	
	inline void  setAlpha (float a) { alpha = a; }
	inline const float getAlpha () const { return alpha; }
	
	inline void  setIndexOfRefraction (float r) { ior = r; }
	inline const float getIndexOfRefraction () const { return ior; }
	
	inline void  setTexture (int mapType, unsigned int t) { texturemap[mapType] = t; has_texture[mapType]=true;}
	inline const unsigned int getTexture (int mapType) const { if (has_texture[mapType]) return texturemap[mapType]; else return INT_MAX; }
};

