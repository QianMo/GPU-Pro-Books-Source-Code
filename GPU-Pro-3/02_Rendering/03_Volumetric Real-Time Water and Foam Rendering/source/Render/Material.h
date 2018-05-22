#ifndef __MATERIAL__H__
#define __MATERIAL__H__

class Material
{
public:
	struct MaterialDefinition 
	{
		bool lighting;

		float ambiente[4];
		float diffuse[4];
		float specular[4];
		float shininess;

		unsigned int textureId;
		unsigned int normalMapId;

		bool useParallaxMapping;
	};
	Material(void);
	~Material(void);

	// inits the material
	void Init(const MaterialDefinition& mat);

	// returns the material
	const MaterialDefinition GetMaterial(void) const { return material; }
	
private:
	// key for material
	unsigned int key;

	// the material
	MaterialDefinition material;
};

#endif