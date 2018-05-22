#include "../Render/RenderObjectKey.h"
#include <assert.h>

// -----------------------------------------------------------------------------
// ---------------- RenderObjectKey::RenderObjectKey ---------------------------
// -----------------------------------------------------------------------------
RenderObjectKey::RenderObjectKey(void)
{
}

// -----------------------------------------------------------------------------
// ---------------- RenderObjectKey::~RenderObjectKey --------------------------
// -----------------------------------------------------------------------------
RenderObjectKey::~RenderObjectKey(void)
{
}

// -----------------------------------------------------------------------------
// ---------------- RenderObjectKey::~RenderObjectKey --------------------------
// -----------------------------------------------------------------------------
void RenderObjectKey::Init(const KeyData& data)
{
	key = data;
	assert(key.materialId <= 32);
}

// -----------------------------------------------------------------------------
// ---------------- RenderObjectKey::~RenderObjectKey --------------------------
// -----------------------------------------------------------------------------
const int RenderObjectKey::GetIntKey(void) const
{
	int returnKey = 0;

	returnKey = (key.useParallaxMapping<<5) | (key.materialId<<0);

	return returnKey;
}