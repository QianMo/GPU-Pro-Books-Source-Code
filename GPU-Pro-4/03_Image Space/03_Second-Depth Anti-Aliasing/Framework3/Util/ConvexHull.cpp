
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

#include "ConvexHull.h"
#include <malloc.h>
#include <stdio.h>

ConvexHull::ConvexHull()
{
	m_Root = NULL;
	m_Curr = NULL;
	m_Count = 0;
}

ConvexHull::~ConvexHull()
{
	Clear();
}

void ConvexHull::Clear()
{
	if (m_Root)
	{
		CHNode *node = m_Root;
		CHNode *next;
		do
		{
			next = node->Next;

			delete node;
			node = next;
		} while (node != m_Root);

		m_Root = NULL;
		m_Count = 0;
	}
	m_Curr = NULL;
}

bool ConvexHull::InsertPoint(const float2 &point)
{
	if (m_Count < 2)
	{
		CHNode *node = new CHNode;
		node->Point = point;

		if (m_Root == NULL)
		{
			m_Root = node;
		}
		else
		{
			node->Prev = m_Root;
			node->Next = m_Root;
		}

		m_Root->Next = node;
		m_Root->Prev = node;
		++m_Count;
		return true;
	}

	CHNode *node = m_Root;

	const float2 &v0 = node->Prev->Point;
	const float2 &v1 = node->Point;

	float2 dir = v1 - v0;
	float2 nrm(-dir.y, dir.x);

	if (dot(point - v0, nrm) > 0)
	{
		do
		{
			node = node->Prev;
			const float2 &v0 = node->Prev->Point;
			const float2 &v1 = node->Point;

			float2 dir = v1 - v0;
			float2 nrm(-dir.y, dir.x);

			if (dot(point - v0, nrm) <= 0)
			{
				node = node->Next;
				break;
			}

		} while (true);
	}
	else
	{
		do
		{
			const float2 &v0 = node->Point;
			node = node->Next;
			const float2 &v1 = node->Point;

			float2 dir = v1 - v0;
			float2 nrm(-dir.y, dir.x);

			if (dot(point - v0, nrm) > 0)
				break;
			if (node == m_Root)
				return false;

		} while (true);
	}

	
	do
	{
		const float2 &v0 = node->Point;
		const float2 &v1 = node->Next->Point;

		float2 dir = v1 - v0;
		float2 nrm(-dir.y, dir.x);

		if (dot(point - v0, nrm) <= 0)
		{
			break;
		}

		// Delete this node
		node->Prev->Next = node->Next;
		node->Next->Prev = node->Prev;

		CHNode *del = node;
		node = node->Next;
		delete del;
		--m_Count;

	} while (true);

	CHNode *new_node = new CHNode;
	new_node->Point = point;
	++m_Count;

	new_node->Prev = node->Prev;
	new_node->Next = node;

	node->Prev->Next = new_node;
	node->Prev = new_node;

	m_Root = new_node;

	return true;
}

struct Line
{
	float2 v;
	float2 d;
};

#define perp(u, v) ((u).x * (v).y - (u).y * (v).x)

bool Intersect(float2 &point, const Line &line0, const Line &line1)
{
#if 0
	float d = perp(line0.d, line1.d);
	if (d > -0.000000000001f) // Parallel lines
		return false;

	float2 diff = line0.v - line1.v;

	float t = perp(line1.d, diff);

	if (t > 0.0f) // Intersects on the wrong side
		return false;

	point = line0.v + (t / d) * line0.d;
	return true;
#else
	float d = perp(line0.d, line1.d);
	if (fabsf(d) < 0.000000000001f) // Parallel lines
		return false;

	float t = perp(line1.d, line0.v - line1.v) / d;

	if (t < 0.5f) // Intersects on the wrong side
		return false;

	point = line0.v + t * line0.d;
	return true;

#endif
}

bool IntersectNoParallelCheck(float2 &point, const Line &line0, const Line &line1)
{
	float d = perp(line0.d, line1.d);
	float t = perp(line1.d, line0.v - line1.v) / d;

	if (t < 0.5f) // Intersects on the wrong side
		return false;

	point = line0.v + t * line0.d;
	return true;
}

float AreaX2Of(const float2 &v0, const float2 &v1, const float2 &v2)
{
	float2 u = v1 - v0;
	float2 v = v2 - v0;

	return /*fabsf*/(u.y * v.x - u.x * v.y);
}

bool ConvexHull::RemoveLeastRelevantEdge()
{
	CHNode *min_node = NULL;
	float2 min_pos;
	float min_area = 1e10f;


	CHNode *node = m_Root;
	do
	{
		const float2 &v0 = node->Prev->Point;
		const float2 &v1 = node->Point;
		const float2 &v2 = node->Next->Point;
		const float2 &v3 = node->Next->Next->Point;

		Line line0 = { v0, v1 - v0 };
		Line line1 = { v2, v3 - v2 };

		float2 v;
		if (IntersectNoParallelCheck(v, line0, line1))
		{
			float area = AreaX2Of(v1, v, v2);
			if (area < min_area)
			{
				min_node = node;
				min_pos = v;
				min_area = area;
			}
		}

		node = node->Next;
	} while (node != m_Root);

	if (min_node)
	{
		min_node->Point = min_pos;

		CHNode *del = min_node->Next;
		min_node->Next->Next->Prev = min_node;
		min_node->Next = min_node->Next->Next;

		if (del == m_Root)
			m_Root = min_node;

		delete del;

		--m_Count;

		return true;
	}

	return false;
}

uint ConvexHull::FindOptimalPolygon(float2 *dest, uint vertex_count, float *area)
{
	if (vertex_count > m_Count)
		vertex_count = m_Count;

	if (vertex_count < 3)
	{
		if (area)
			*area = 0.0f;
		return 0;
	}

	if (vertex_count > 8)
		vertex_count = 8;

	// Allocate memory on stack
	Line *lines = (Line *) alloca(m_Count * sizeof(Line));
	//Line *lines = (Line *) (intptr_t(alloca(m_Count * sizeof(Line) + 64)) & ~intptr_t(63));

	CHNode *node = m_Root;

	// Precompute lines
	uint n = 0;
	do
	{
		lines[n].v = node->Point;
		lines[n].d = node->Next->Point - node->Point;

		// Move origin to center of line
		//lines[n].v += 0.5f * lines[n].d;

		node = node->Next;
		++n;
	} while (node != m_Root);

	ASSERT(n == m_Count);




	float min_area = 1e10f;

	//float2 v0, v1, v2, v3, v4, v5, v6, v7;
	/*__declspec(align(16))*/ float2 v[8];
	float2 &v0 = v[0];
	float2 &v1 = v[1];
	float2 &v2 = v[2];
	float2 &v3 = v[3];
	float2 &v4 = v[4];
	float2 &v5 = v[5];
	float2 &v6 = v[6];
	float2 &v7 = v[7];

	assume(n > 0);

	// This can probably be made a lot prettier and generic
	switch (vertex_count)
	{
	case 3:
		for (uint x = 0; x < n; x++)
		{
			for (uint y = x + 1; y < n; y++)
			{
				if (Intersect(v0, lines[x], lines[y]))
				{
					for (uint z = y + 1; z < n; z++)
					{
						if (Intersect(v1, lines[y], lines[z]))
						{
							if (Intersect(v2, lines[z], lines[x]))
							{
								float2 u0 = v1 - v0;
								float2 u1 = v2 - v0;

								float area = (u0.y * u1.x - u0.x * u1.y);

								if (area < min_area)
								{
									min_area = area;
									dest[0] = v0;
									dest[1] = v1;
									dest[2] = v2;
								}
							}
						}
					}
				}
			}
		}
		break;
	case 4:
		for (uint x = 0; x < n; x++)
		{
			for (uint y = x + 1; y < n; y++)
			{
				if (Intersect(v0, lines[x], lines[y]))
				{
					for (uint z = y + 1; z < n; z++)
					{
						if (Intersect(v1, lines[y], lines[z]))
						{
							for (uint w = z + 1; w < n; w++)
							{
								if (Intersect(v2, lines[z], lines[w]))
								{
									if (Intersect(v3, lines[w], lines[x]))
									{
										float2 u0 = v1 - v0;
										float2 u1 = v2 - v0;
										float2 u2 = v3 - v0;

										float area = 
											(u0.y * u1.x - u0.x * u1.y) +
											(u1.y * u2.x - u1.x * u2.y);

										if (area < min_area)
										{
											min_area = area;
											dest[0] = v0;
											dest[1] = v1;
											dest[2] = v2;
											dest[3] = v3;
										}
									}
								}
							}
						}
					}
				}
			}
		}
		break;
	case 5:
		for (uint x = 0; x < n; x++)
		{
			for (uint y = x + 1; y < n; y++)
			{
				if (Intersect(v0, lines[x], lines[y]))
				{
					for (uint z = y + 1; z < n; z++)
					{
						if (Intersect(v1, lines[y], lines[z]))
						{
							for (uint w = z + 1; w < n; w++)
							{
								if (Intersect(v2, lines[z], lines[w]))
								{
									for (uint r = w + 1; r < n; r++)
									{
										if (Intersect(v3, lines[w], lines[r]))
										{
											if (Intersect(v4, lines[r], lines[x]))
											{
												float2 u0 = v1 - v0;
												float2 u1 = v2 - v0;
												float2 u2 = v3 - v0;
												float2 u3 = v4 - v0;

												float area = 
													(u0.y * u1.x - u0.x * u1.y) +
													(u1.y * u2.x - u1.x * u2.y) +
													(u2.y * u3.x - u2.x * u3.y);

												if (area < min_area)
												{
													min_area = area;
													dest[0] = v0;
													dest[1] = v1;
													dest[2] = v2;
													dest[3] = v3;
													dest[4] = v4;
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
		break;
	case 6:
		for (uint x = 0; x < n; x++)
		{
			for (uint y = x + 1; y < n; y++)
			{
				if (Intersect(v0, lines[x], lines[y]))
				{
					for (uint z = y + 1; z < n; z++)
					{
						if (Intersect(v1, lines[y], lines[z]))
						{
							for (uint w = z + 1; w < n; w++)
							{
								if (Intersect(v2, lines[z], lines[w]))
								{
									for (uint r = w + 1; r < n; r++)
									{
										if (Intersect(v3, lines[w], lines[r]))
										{
											for (uint s = r + 1; s < n; s++)
											{
												if (Intersect(v4, lines[r], lines[s]))
												{
													if (Intersect(v5, lines[s], lines[x]))
													{
														float2 u0 = v1 - v0;
														float2 u1 = v2 - v0;
														float2 u2 = v3 - v0;
														float2 u3 = v4 - v0;
														float2 u4 = v5 - v0;

														float area = 
															(u0.y * u1.x - u0.x * u1.y) +
															(u1.y * u2.x - u1.x * u2.y) +
															(u2.y * u3.x - u2.x * u3.y) +
															(u3.y * u4.x - u3.x * u4.y);

														if (area < min_area)
														{
															min_area = area;
															dest[0] = v0;
															dest[1] = v1;
															dest[2] = v2;
															dest[3] = v3;
															dest[4] = v4;
															dest[5] = v5;
														}
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
		break;
	case 7:
		for (uint x = 0; x < n; x++)
		{
			for (uint y = x + 1; y < n; y++)
			{
				if (Intersect(v0, lines[x], lines[y]))
				{
					for (uint z = y + 1; z < n; z++)
					{
						if (Intersect(v1, lines[y], lines[z]))
						{
							for (uint w = z + 1; w < n; w++)
							{
								if (Intersect(v2, lines[z], lines[w]))
								{
									for (uint r = w + 1; r < n; r++)
									{
										if (Intersect(v3, lines[w], lines[r]))
										{
											for (uint s = r + 1; s < n; s++)
											{
												if (Intersect(v4, lines[r], lines[s]))
												{
													for (uint t = s + 1; t < n; t++)
													{
														if (Intersect(v5, lines[s], lines[t]))
														{
															if (Intersect(v6, lines[t], lines[x]))
															{
																float2 u0 = v1 - v0;
																float2 u1 = v2 - v0;
																float2 u2 = v3 - v0;
																float2 u3 = v4 - v0;
																float2 u4 = v5 - v0;
																float2 u5 = v6 - v0;

																float area = 
																	(u0.y * u1.x - u0.x * u1.y) +
																	(u1.y * u2.x - u1.x * u2.y) +
																	(u2.y * u3.x - u2.x * u3.y) +
																	(u3.y * u4.x - u3.x * u4.y) +
																	(u4.y * u5.x - u4.x * u5.y);

																if (area < min_area)
																{
																	min_area = area;
																	dest[0] = v0;
																	dest[1] = v1;
																	dest[2] = v2;
																	dest[3] = v3;
																	dest[4] = v4;
																	dest[5] = v5;
																	dest[6] = v6;
																}
															}
														}
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
		break;
	case 8:
		for (uint x = 0; x < n; x++)
		{
			for (uint y = x + 1; y < n; y++)
			{
				if (Intersect(v0, lines[x], lines[y]))
				{
					for (uint z = y + 1; z < n; z++)
					{
						if (Intersect(v1, lines[y], lines[z]))
						{
							for (uint w = z + 1; w < n; w++)
							{
								if (Intersect(v2, lines[z], lines[w]))
								{
									for (uint r = w + 1; r < n; r++)
									{
										if (Intersect(v3, lines[w], lines[r]))
										{
											for (uint s = r + 1; s < n; s++)
											{
												if (Intersect(v4, lines[r], lines[s]))
												{
													for (uint t = s + 1; t < n; t++)
													{
														if (Intersect(v5, lines[s], lines[t]))
														{
															for (uint u = t + 1; u < n; u++)
															{
																if (Intersect(v6, lines[t], lines[u]))
																{
																	if (Intersect(v7, lines[u], lines[x]))
																	{
																		float2 u0 = v1 - v0;
																		float2 u1 = v2 - v0;
																		float2 u2 = v3 - v0;
																		float2 u3 = v4 - v0;
																		float2 u4 = v5 - v0;
																		float2 u5 = v6 - v0;
																		float2 u6 = v7 - v0;

																		float area = 
																			(u0.y * u1.x - u0.x * u1.y) +
																			(u1.y * u2.x - u1.x * u2.y) +
																			(u2.y * u3.x - u2.x * u3.y) +
																			(u3.y * u4.x - u3.x * u4.y) +
																			(u4.y * u5.x - u4.x * u5.y) +
																			(u5.y * u6.x - u5.x * u6.y);

																		if (area < min_area)
																		{
																			min_area = area;
																			dest[0] = v0;
																			dest[1] = v1;
																			dest[2] = v2;
																			dest[3] = v3;
																			dest[4] = v4;
																			dest[5] = v5;
																			dest[6] = v6;
																			dest[7] = v7;
																		}
																	}
																}
															}
														}
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
		break;
	}

	if (area != NULL)
	{
		*area = 0.5f * min_area;
	}

	return vertex_count;
}

float ConvexHull::GetArea() const
{
	if (m_Count < 3)
		return 0.0f;

	float area = 0.0f;

	const float2 &v0 = m_Root->Point;

	CHNode *node = m_Root->Next;
	do
	{
		const float2 &v1 = node->Point;
		node = node->Next;
		const float2 &v2 = node->Point;

		area += AreaX2Of(v0, v1, v2);

	} while (node != m_Root);

	return 0.5f * area;
}
