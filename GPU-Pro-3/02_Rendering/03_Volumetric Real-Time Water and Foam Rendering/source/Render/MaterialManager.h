#ifndef __MATERIALMANAGER__H__
#define __MATERIALMANAGER__H__

#include <map>
#include <list>
#include <string>
using namespace std;

#include "../Util/Singleton.h"
#include "../Render/Material.h"
#include "../XMLParser/XmlNotify.h"

class MaterialManager : public Singleton<MaterialManager>, XmlNotify
{
	friend class Singleton<MaterialManager>;

public:
	MaterialManager(void);
	~MaterialManager(void);

	// loads the materials from a spec. file
	void LoadMaterialsFromFile(const char* fileName);

	// adds a material to the map
	const int AddMaterial(const Material& mat);

	// sets a material for rendering
	void SetMaterial(const int& key);

	// returns the material for the given key
	Material::MaterialDefinition GetMaterial(const int& key);

	// checks if a materials key existst or not
	bool CheckMaterial(const int& key);

	// exits the material manager
	void Exit(void);

	// callback functions for xml-parsing
	void foundNode		(string& name, string& attributes);
	void foundElement	(string& name, string& value, string& attributes);
	void startElement	(string& name, string& value, string& attributes);
	void endElement		(string& name, string& value, string& attributes);

private:
	void AddCurrentMaterial(void);

	// the stored materails
	map<int, Material> materials;
	map<int, Material>::iterator iter;

	enum NODETYPES
	{
		NODETYPE_NONE = 0,
		NODETYPE_LIGHTING,
		NODETYPE_MATERIAL,
		NODETYPE_AMBIENTE,
		NODETYPE_DIFFUSE,
		NODETYPE_SPECULAR,
		NODETYPE_SHININESS,
		NODETYPE_TEXTUREFILENAME,
		NODETYPE_NORMALMAPFILENAME,
	};
	struct loadMaterial
	{
		bool lighting;
		float ambiente[4];
		float diffuse[4];
		float specular[4];
		float shininess;
		unsigned int texId;
		unsigned int normalMapId;
		bool useParallaxMapping;
	};

	// materials loaded?
	bool isInit;

	NODETYPES currentNode;
	loadMaterial currentMaterial;
};

#endif