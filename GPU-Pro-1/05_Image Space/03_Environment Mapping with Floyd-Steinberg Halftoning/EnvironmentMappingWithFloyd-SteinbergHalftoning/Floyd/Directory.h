#pragma once

#include "Role.h"

template<class _Kty, class _Ty, class _Pr=std::less<_Kty> >
class CompositMap : public std::map<_Kty, _Ty, _Pr>
{
public:
	void deleteAll()
	{
		iterator i = begin();
		iterator e = end();
		while(i != e)
		{
			if(i->second)
				delete i->second;
			i++;
		}
	}
};

template<class _Kty, class _Ty>
class CompositMapList : public std::map<_Kty, _Ty>
{
public:
	void deleteAll()
	{
		iterator i = begin();
		iterator e = end();
		while(i != e)
		{
			i->second.deleteAll();
			i++;
		}
	}
};

template<class _Ty>
class CompositList : public std::vector<_Ty>
{
public:
	void deleteAll()
	{
		iterator i = begin();
		iterator e = end();
		while(i != e)
		{
			if(*i)
				delete *i;
			i++;
		}
	}
};

template<class _Kty, class _Ty>
class ResourceMap : public std::map<_Kty, _Ty>
{
public:
	void releaseAll()
	{
		iterator i = begin();
		iterator e = end();
		while(i != e)
		{
			if(i->second)
				i->second->Release();
			i++;
		}
	}
};

struct ComparePointers
{
	bool operator() (const void* const& a, const void* const& b) const
	{
		return a < b;
	}
};

struct CompareRoles
{
	bool operator() (const Role& a, const Role& b) const
	{
		return a < b;
	}
};

struct CompareShorts
{
	bool operator() (unsigned short a, unsigned short b) const
	{
		return a < b;
	}
};


/// Associative container class for D3D texture references.
typedef ResourceMap<const std::wstring, ID3D10ShaderResourceView*> ShaderResourceViewDirectory;
/// Associative container class for D3D vectors.
typedef std::map<const std::wstring, D3DXVECTOR4>	VectorDirectory;

/// Associative container class for D3D mesh references.
typedef ResourceMap<const std::wstring, ID3DX10Mesh*>     MeshDirectory;

class Entity;
/// Associative container class for Entity references.
typedef std::map<const std::wstring, Entity*>	  EntityDirectory;

class Repertoire;
typedef CompositMap<const std::wstring, Repertoire*>  RepertoireDirectory;

class ShadedMesh;
/// Associative container class for ShadedMesh references.
typedef CompositMap<const std::wstring, ShadedMesh*>	  ShadedMeshDirectory;

typedef std::map<const std::wstring, const Role>	RoleDirectory;

class Camera;
/// Associative container class for Camera references.
typedef CompositMap<const std::wstring, Camera*>	  CameraDirectory;

class PhysicsMaterial;
/// Associative container class for NxMaterial references.
typedef CompositMap<const std::wstring, PhysicsMaterial*>	PhysicsMaterialDirectory;

class PhysicsModel;
/// Associative container class for PhysicsModel references.
typedef CompositMap<const std::wstring, PhysicsModel*>	PhysicsModelDirectory;

class PhysicsEntityWheel;
typedef CompositMap<const std::wstring, PhysicsEntityWheel*> PhysicsEntityWheelDirectory;

class PhysicsController;
typedef CompositMap<const std::wstring, PhysicsController*> PhysicsControllerDirectory;

class WheelController;
typedef CompositMap<const std::wstring, WheelController*> WheelControllerDirectory;

class Cueable;
typedef CompositMap<const std::wstring, Cueable*> CueableDirectory;

class Act;
typedef CompositMap<const std::wstring, Act*> ActDirectory;

typedef std::map<unsigned short, unsigned short, CompareShorts> IndexMap;

typedef std::map<const std::wstring, NxHeightField*>	PhysicsHeightFieldDirectory;

class CharacterModel;
typedef CompositMap<const std::wstring, CharacterModel*>	CharacterModelDirectory;

class CharacterBone;
typedef CompositMap<const std::wstring, CharacterBone*>	CharacterBoneDirectory;
typedef CompositList<CharacterBone*>	CharacterBoneList;

typedef std::vector<NxActor*>	PhysicsActorList;
typedef std::vector<NxJoint*>	PhysicsJointList;

class Clip;
typedef CompositMap<const std::wstring, Clip*>	ClipDirectory;

typedef ResourceMap<const std::wstring, ID3D10Effect*>	EffectDirectory;

class SceneManager;
typedef std::vector<SceneManager*>	SceneManagerList;