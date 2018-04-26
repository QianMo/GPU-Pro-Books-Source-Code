#include "DXUT.h"
#include "Repertoire.h"
#include "Rendition.h"
#include "RenderContext.h"

Repertoire::Repertoire(void)
{
}

Repertoire::~Repertoire(void)
{
	renditions.deleteAll();
}

void Repertoire::addRendition(const Role role, Rendition* rendition)
{
	renditions[role] = rendition;
}

Rendition* Repertoire::getRendition(const Role role)
{
	RenditionDirectory::iterator i = renditions.find(role);
	if(i == renditions.end())
		return NULL;
	return i->second;
}