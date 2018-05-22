
/* * * * * * * * * * * * * Author's note * * * * * * * * * * * *\
*   _       _   _       _   _       _   _       _     _ _ _ _   *
*  |_|     |_| |_|     |_| |_|_   _|_| |_|     |_|  _|_|_|_|_|  *
*  |_|_ _ _|_| |_|     |_| |_|_|_|_|_| |_|     |_| |_|_ _ _     *
*  |_|_|_|_|_| |_|     |_| |_| |_| |_| |_|     |_|   |_|_|_|_   *
*  |_|     |_| |_|_ _ _|_| |_|     |_| |_|_ _ _|_|  _ _ _ _|_|  *
*  |_|     |_|   |_|_|_|   |_|     |_|   |_|_|_|   |_|_|_|_|    *
*                                                               *
*                     http://www.humus.name                     *
*                                                                *
* This file is a part of the work done by Humus. You are free to   *
* use the code in any way you like, modified, unmodified or copied   *
* into your own work. However, I expect you to respect these points:  *
*  - If you use this file and its contents unmodified, or use a major *
*    part of this file, please credit the author and leave this note. *
*  - For use in anything commercial, please request my approval.     *
*  - Share your work and ideas too as much as you can.             *
*                                                                *
\* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef _CONVEXHULL_H_
#define _CONVEXHULL_H_

#include "../Platform.h"
#include "../Math/Vector.h"

struct CHNode
{
	CHNode *Prev;
	CHNode *Next;
	float2 Point;
};

class ConvexHull
{
public:
	ConvexHull();
	~ConvexHull();

	void Clear();
	bool InsertPoint(const float2 &point);
	bool RemoveLeastRelevantEdge();
	uint FindOptimalPolygon(float2 *dest, uint vertex_count, float *area = NULL);

	bool GoToFirst() { return (m_Curr = m_Root) != NULL; }
	bool GoToNext () { return (m_Curr = m_Curr->Next) != m_Root; }

	const float2 &GetCurrPoint() const { return m_Curr->Point; }
	const float2 &GetNextPoint() const { return m_Curr->Next->Point; }
	const float2 &GetPrevPoint() const { return m_Curr->Prev->Point; }

	uint GetCount() const { return m_Count; }
	float GetArea() const;

protected:
	CHNode *m_Root;
	CHNode *m_Curr;
	uint m_Count;

};

#endif // _CONVEXHULL_H_
