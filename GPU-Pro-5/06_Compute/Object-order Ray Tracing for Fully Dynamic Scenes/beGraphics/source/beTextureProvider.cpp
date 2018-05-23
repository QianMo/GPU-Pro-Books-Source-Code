/*****************************************************/
/* breeze Engine Graphics Module  (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beGraphicsInternal/stdafx.h"
#include "beGraphics/beTextureProvider.h"

#include <lean/functional/predicates.h>

namespace beGraphics
{

// Transfers all textures from the given source texture provider to the given destination texture provider.
void TransferTextures(TextureProvider &dest, const TextureProvider &source)
{
	const uint4 count = dest.GetTextureCount();
	const uint4 srcCount = source.GetTextureCount();

	uint4 nextID = 0;

	for (uint4 srcID = 0; srcID < srcCount; ++srcID)
	{
		utf8_ntr srcName = source.GetTextureName(srcID);

		uint4 lowerID = nextID;
		uint4 upperID = nextID;

		for (uint4 i = 0; i < count; ++i)
		{
			// Perform bi-directional search: even == forward; odd == backward
			uint4 id = (lean::is_odd(i) | (upperID == count)) & (lowerID != 0)
				? --lowerID
				: upperID++;

			if (dest.GetTextureName(id) == srcName)
			{
				dest.SetTexture(id, source.GetTexture(srcID));

				// Start next search with next texture
				nextID = id + 1;
				break;
			}
		}
	}
}

} // namespace
