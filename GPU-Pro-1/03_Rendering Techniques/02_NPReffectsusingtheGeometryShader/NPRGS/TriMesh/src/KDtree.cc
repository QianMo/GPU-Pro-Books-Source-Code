/*
Szymon Rusinkiewicz
Princeton University

KDtree.cc
A K-D tree for points, with limited capabilities (find nearest point to
a given point, or to a ray).
*/

#include <cmath>
#include <string.h>
#include "KDtree.h"
#include "mempool.h"
#include <vector>
#include <algorithm>
using std::vector;
using std::swap;
using std::sqrt;


// Small utility fcns
static inline float sqr(float x)
{
	return x*x;
}

static inline float dist2(const float *x, const float *y)
{
	return sqr(x[0]-y[0]) + sqr(x[1]-y[1]) + sqr(x[2]-y[2]);
}

static inline float dist2ray2(const float *x, const float *p, const float *d)
{
	float xp0 = x[0]-p[0], xp1 = x[1]-p[1], xp2 = x[2]-p[2];
	return sqr(xp0) + sqr(xp1) + sqr(xp2) -
	       sqr(xp0*d[0] + xp1*d[1] + xp2*d[2]);
}


// Class for nodes in the K-D tree
class KDtree::Node {
private:
	static PoolAlloc memPool;

public:
	// A place to put all the stuff required while traversing the K-D
	// tree, so we don't have to pass tons of variables at each fcn call
	struct Traversal_Info {
		const float *p, *dir;
		const float *closest;
		float closest_d, closest_d2;
		const KDtree::CompatFunc *iscompat;
	};

	enum { MAX_PTS_PER_NODE = 7 };


	// The node itself

	int npts; // If this is 0, intermediate node.  If nonzero, leaf.

	union {
		struct {
			float center[3];
			float r;
			int splitaxis;
			Node *child1, *child2;
		} node;
		struct {
			const float *p[MAX_PTS_PER_NODE];
		} leaf;
	};

	Node(const float **pts, int n);
	~Node();

	void find_closest_to_pt(Traversal_Info &k) const;
	void find_closest_to_ray(Traversal_Info &k) const;

	void *operator new(size_t n) { return memPool.alloc(n); }
	void operator delete(void *p, size_t n) { memPool.free(p,n); }
};


// Class static variable
PoolAlloc KDtree::Node::memPool(sizeof(KDtree::Node));


// Create a KD tree from the points pointed to by the array pts
KDtree::Node::Node(const float **pts, int n)
{
	// Leaf nodes
	if (n <= MAX_PTS_PER_NODE) {
		npts = n;
		memcpy(leaf.p, pts, n * sizeof(float *));
		return;
	}


	// Else, interior nodes
	npts = 0;

	// Find bbox
	float xmin = pts[0][0], xmax = pts[0][0];
	float ymin = pts[0][1], ymax = pts[0][1];
	float zmin = pts[0][2], zmax = pts[0][2];
	for (int i = 1; i < n; i++) {
		if (pts[i][0] < xmin)  xmin = pts[i][0];
		if (pts[i][0] > xmax)  xmax = pts[i][0];
		if (pts[i][1] < ymin)  ymin = pts[i][1];
		if (pts[i][1] > ymax)  ymax = pts[i][1];
		if (pts[i][2] < zmin)  zmin = pts[i][2];
		if (pts[i][2] > zmax)  zmax = pts[i][2];
	}

	// Find node center and size
	node.center[0] = 0.5f * (xmin+xmax);
	node.center[1] = 0.5f * (ymin+ymax);
	node.center[2] = 0.5f * (zmin+zmax);
	float dx = xmax-xmin;
	float dy = ymax-ymin;
	float dz = zmax-zmin;
	node.r = 0.5f * sqrt(sqr(dx) + sqr(dy) + sqr(dz));

	// Find longest axis
	node.splitaxis = 2;
	if (dx > dy) {
		if (dx > dz)
			node.splitaxis = 0;
	} else {
		if (dy > dz)
			node.splitaxis = 1;
	}

	// Partition
	const float splitval = node.center[node.splitaxis];
	const float **left = pts, **right = pts + n - 1;
	while (1) {
		while ((*left)[node.splitaxis] < splitval)
			left++;
		while ((*right)[node.splitaxis] >= splitval)
			right--;
		if (right < left)
			break;
		swap(*left, *right);
		left++; right--;
	}


	// Build subtrees
	node.child1 = new Node(pts, left-pts);
	node.child2 = new Node(left, n-(left-pts));
}


// Destroy a KD tree node
KDtree::Node::~Node()
{
	if (!npts) {
		delete node.child1;
		delete node.child2;
	}
}


// Crawl the KD tree
void KDtree::Node::find_closest_to_pt(KDtree::Node::Traversal_Info &k) const
{
	// Leaf nodes
	if (npts) {
		for (int i = 0; i < npts; i++) {
			float myd2 = dist2(leaf.p[i], k.p);
			if ((myd2 < k.closest_d2) &&
			    (!k.iscompat || (*k.iscompat)(leaf.p[i]))) {
				k.closest_d2 = myd2;
				k.closest_d = sqrt(k.closest_d2);
				k.closest = leaf.p[i];
			}
		}
		return;
	}


	// Check whether to abort
	if (dist2(node.center, k.p) >= sqr(node.r + k.closest_d))
		return;

	// Recursive case
	float myd = node.center[node.splitaxis] - k.p[node.splitaxis];
	if (myd >= 0.0f) {
		node.child1->find_closest_to_pt(k);
		if (myd < k.closest_d)
			node.child2->find_closest_to_pt(k);
	} else {
		node.child2->find_closest_to_pt(k);
		if (-myd < k.closest_d)
			node.child1->find_closest_to_pt(k);
	}
}


// Crawl the KD tree to look for the closest point to
// the line going through k.p in the direction k.dir
void KDtree::Node::find_closest_to_ray(KDtree::Node::Traversal_Info &k) const
{
	// Leaf nodes
	if (npts) {
		for (int i = 0; i < npts; i++) {
			float myd2 = dist2ray2(leaf.p[i], k.p, k.dir);
			if ((myd2 < k.closest_d2) &&
			    (!k.iscompat || (*k.iscompat)(leaf.p[i]))) {
				k.closest_d2 = myd2;
				k.closest_d = sqrt(k.closest_d2);
				k.closest = leaf.p[i];
			}
		}
		return;
	}


	// Check whether to abort
	if (dist2ray2(node.center, k.p, k.dir) >= sqr(node.r + k.closest_d))
		return;

	// Recursive case
	if (k.p[node.splitaxis] < node.center[node.splitaxis] ) {
		node.child1->find_closest_to_ray(k);
		node.child2->find_closest_to_ray(k);
	} else {
		node.child2->find_closest_to_ray(k);
		node.child1->find_closest_to_ray(k);
	}
}


// Create a KDtree from a list of points (i.e., ptlist is a list of 3*n floats)
void KDtree::build(const float *ptlist, int n)
{
	vector<const float *> pts(n);
	for (int i = 0; i < n; i++)
		pts[i] = ptlist + i * 3;

	root = new Node(&(pts[0]), n);
}


// Delete a KDtree
KDtree::~KDtree()
{
	delete root;
}


// Return the closest point in the KD tree to p
const float *KDtree::closest_to_pt(const float *p, float maxdist2,
				   const CompatFunc *iscompat /* = NULL */) const
{
	Node::Traversal_Info k;

	k.p = p;
	k.iscompat = iscompat;
	k.closest = NULL;
	if (maxdist2 <= 0.0f)
		maxdist2 = sqr(root->node.r);
	k.closest_d2 = maxdist2;
	k.closest_d = sqrt(k.closest_d2);

	root->find_closest_to_pt(k);

	return k.closest;
}


// Return the closest point in the KD tree to the line
// going through p in the direction dir
const float *KDtree::closest_to_ray(const float *p, const float *dir,
				    float maxdist2,
				    const CompatFunc *iscompat /* = NULL */) const
{
	Node::Traversal_Info k;

	float one_over_dir_len = 1.0f / sqrt(sqr(dir[0])+sqr(dir[1])+sqr(dir[2]));
	float normalized_dir[3] = { dir[0] * one_over_dir_len, 
				    dir[1] * one_over_dir_len, 
				    dir[2] * one_over_dir_len };
	k.dir = normalized_dir;
	k.p = p;
	k.iscompat = iscompat;
	k.closest = NULL;
	if (maxdist2 <= 0.0f)
		maxdist2 = sqr(root->node.r);
	k.closest_d2 = maxdist2;
	k.closest_d = sqrt(k.closest_d2);

	root->find_closest_to_ray(k);

	return k.closest;
}

