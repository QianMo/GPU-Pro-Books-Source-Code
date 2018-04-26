#pragma once

#include "Role.h"
#include "Directory.h"
class Rendition;
class RenderContext;

class Repertoire
{
public:
	/// Associative container class for Rendition references. Used in ShadedMesh.
	typedef CompositMap<const Role, Rendition*, CompareRoles>	  RenditionDirectory;
private:
	RenditionDirectory renditions;

public:

	Repertoire(void);
	~Repertoire(void);

	void addRendition(const Role role, Rendition* rendition);

	Rendition* getRendition(const Role role);

	RenditionDirectory::iterator begin(){return renditions.begin();}
	RenditionDirectory::iterator end(){return renditions.end();}
};
