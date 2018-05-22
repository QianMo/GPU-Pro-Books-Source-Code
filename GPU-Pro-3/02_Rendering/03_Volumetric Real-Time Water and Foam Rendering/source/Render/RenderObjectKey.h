#ifndef __RENDEROBJECTKEY__H__
#define __RENDEROBJECTKEY__H__

class RenderObjectKey
{
public:
	struct KeyData 
	{
		int materialId;
		bool useParallaxMapping;
	};
	RenderObjectKey(void);
	~RenderObjectKey(void);

	// init the key
	void Init(const KeyData& data);

	// returns the keydata coded as int value
	const int GetIntKey(void) const;

private:
	// key data
	KeyData key;
};

#endif