// ltree.glsl - public domain linear tree functions for GLSL
// author: Jonathan Dupuy (jdupuy@liris.cnrs.fr)
#ifndef LTREE_GLSL
#define LTREE_GLSL

#if (__VERSION__ < 150) // check version
#	error Uncompatible GLSL version
#endif

/* encode / decode */
void lt_encode_2_15(in uint level, in uvec2 p, out uint key);
void lt_encode_2_30(in uint level, in uvec2 p, out uvec2 key);
void lt_encode_3_9(in uint level, in uvec3 p, out uint key);
void lt_encode_3_19(in uint level, in uvec3 p, out uvec2 key);
void lt_decode_2_15(in uint key, out uint level, out uvec2 p);
void lt_decode_2_30(in uvec2 key, out uint level, out uvec2 p);
void lt_decode_3_9(in uint key, out uint level, out uvec3 p);
void lt_decode_3_19(in uvec2 key, out uint level, out uvec3 p);

/* children nodes */
void lt_children_2_15(in uint key, out uint children[4]);
void lt_children_2_30(in uvec2 key, out uvec2 children[4]);
void lt_children_3_9(in uint key, out uint children[8]);
void lt_children_3_19(in uint key, out uvec2 children[8]);

/* parent node */
uint lt_parent_2_15(in uint key);
uvec2 lt_parent_2_30(in uvec2 key);
uint lt_parent_3_9(in uint key);
uvec2 lt_parent_3_19(in uint key);

/* check if the node is a leaf */
bool lt_is_leaf_2_15(in uint key);
bool lt_is_leaf_2_30(in uvec2 key);
bool lt_is_leaf_3_9(in uint key);
bool lt_is_leaf_3_19(in uvec2 key);

/* check if the node is the root */
bool lt_is_root_2_15(in uint key);
bool lt_is_root_2_30(in uvec2 key);
bool lt_is_root_3_9(in uint key);
bool lt_is_root_3_19(in uvec2 key);

/* check node orientation */
bool lt_is_left_2_15(in uint key);
bool lt_is_right_2_15(in uint key);
bool lt_is_upper_2_15(in uint key);
bool lt_is_lower_2_15(in uint key);

bool lt_is_left_2_30(in uvec2 key);
bool lt_is_right_2_30(in uvec2 key);
bool lt_is_upper_2_30(in uvec2 key);
bool lt_is_lower_2_30(in uvec2 key);

bool lt_is_left_3_9(in uint key);
bool lt_is_right_3_9(in uint key);
bool lt_is_upper_3_9(in uint key);
bool lt_is_lower_3_9(in uint key);
bool lt_is_front_3_9(in uint key);
bool lt_is_back_3_9(in uint key);

bool lt_is_left_3_19(in uvec2 key);
bool lt_is_right_3_19(in uvec2 key);
bool lt_is_upper_3_19(in uvec2 key);
bool lt_is_lower_3_19(in uvec2 key);
bool lt_is_front_3_19(in uvec2 key);
bool lt_is_back_3_19(in uvec2 key);

/* check node orientation (ctd) */
bool lt_is_upper_left_2_15(in uint key);
bool lt_is_upper_right_2_15(in uint key);
bool lt_is_lower_left_2_15(in uint key);
bool lt_is_lower_right_2_15(in uint key);

bool lt_is_upper_left_2_30(in uvec2 key);
bool lt_is_upper_right_2_30(in uvec2 key);
bool lt_is_lower_left_2_30(in uvec2 key);
bool lt_is_lower_right_2_30(in uvec2 key);

bool lt_is_front_upper_left_3_9(in uint key);
bool lt_is_front_upper_right_3_9(in uint key);
bool lt_is_front_lower_left_3_9(in uint key);
bool lt_is_front_lower_right_3_9(in uint key);
bool lt_is_back_upper_left_3_9(in uint key);
bool lt_is_back_upper_right_3_9(in uint key);
bool lt_is_back_lower_left_3_9(in uint key);
bool lt_is_back_lower_right_3_9(in uint key);

bool lt_is_front_upper_left_3_19(in uvec2 key);
bool lt_is_front_upper_right_3_19(in uvec2 key);
bool lt_is_front_lower_left_3_19(in uvec2 key);
bool lt_is_front_lower_right_3_19(in uvec2 key);
bool lt_is_back_upper_left_3_19(in uvec2 key);
bool lt_is_back_upper_right_3_19(in uvec2 key);
bool lt_is_back_lower_left_3_19(in uvec2 key);
bool lt_is_back_lower_right_3_19(in uvec2 key);

/* get normalized position + optionally parent (in [0,1]) */
void lt_cell_2_15(in uint key, out vec2 cell, out float cell_size);
void lt_cell_2_15(in uint key, out vec2 cell, out float cell_size, out vec2 parent_cell);

void lt_cell_2_30(in uvec2 key, out vec2 cell, out float cell_size);
void lt_cell_2_30(in uvec2 key, out vec2 cell, out float cell_size, out vec2 parent_cell);

void lt_cell_3_9(in uint key, out vec3 cell, out float cell_size);
void lt_cell_3_9(in uint key, out vec3 cell, out float cell_size, out vec3 parent_cell);

void lt_cell_3_19(in uvec2 key, out vec3 cell, out float cell_size);
void lt_cell_3_19(in uvec2 key, out vec3 cell, out float cell_size, out vec3 parent_cell);

/*
2tree representation at level 2
===============================

(0,0)                      (0,1)
     +----+----++----+----+
     | 00 | 01 || 04 | 05 |
     +----+----++----+----+
     | 02 | 03 || 06 | 07 |
     +====+====++====+====+
     | 08 | 09 || 12 | 13 |
     +----+----++----+----+
     | 10 | 11 || 14 | 15 |
     +----+----++----+----+
(0,1)                      (1,1)

The representation should minimize the 
number of operations during:
1. decoding: convert from morton to "regular" coordinates
2. splitting: computing children nodes
3. merging: computing the parent node

Splitting (resp. merging) requires at least an increment (resp. decrement),
a logical or and a bitshift, while the decoding requires at least a bit 
deinterleaving procedure.

8bit 2tree split:
k:= ---- --01
--> ---- 0001 <=> ((l+1u) | (0x00 | (n >> 2u))
--> ---- 0101 <=> ((l+1u) | (0x10 | (n >> 2u))
--> ---- 1001 <=> ((l+1u) | (0x20 | (n >> 2u))
--> ---- 1101 <=> ((l+1u) | (0x30 | (n >> 2u))

8bit 2tree merge
---- 0001
-> ---- --00 <=> ((l-1u) | ((n << 2u) & 0xF))

The level is stored in the least significant bits
The active node data is stored after the level
         msb                                 lsb
level 0: ---- ---- ---- ---- ---- ---- ---- 0000
level 1: ---- ---- ---- ---- ---- ---- --nn 0001
level 2: ---- ---- ---- ---- ---- ---- nnmm 0010
level 3: ---- ---- ---- ---- ---- --nn mmoo 0011
etc.
*********************************************/

/*
Dilate a 16bit uint
====================
See "Converting to and from Dilated Integers"
from Rajeev Raman and David S. Wise
*********************************************/
uint lt_dilate_2(in uint x) {
	x = (x | (x << 8u)) & 0x00FF00FF;
	x = (x | (x << 4u)) & 0x0F0F0F0F;
	x = (x | (x << 2u)) & 0x33333333;
	x = (x | (x << 1u)) & 0x55555555;
	return x;
}
uint lt_dilate_3(in uint x) {
	x = (x * 0x10001) & 0xFF0000FF;
	x = (x * 0x00101) & 0x0F00F00F;
	x = (x * 0x00011) & 0xC30C30C3;
	x = (x * 0x00005) & 0x49249249;
	return x;
}

/*
Undilate a 16bit integer
========================
See "Converting to and from Dilated Integers"
from Rajeev Raman and David S. Wise
*********************************************/
uint lt_undilate_2(in uint x) {
	x = (x | (x >> 1u)) & 0x33333333;
	x = (x | (x >> 2u)) & 0x0F0F0F0F;
	x = (x | (x >> 4u)) & 0x00FF00FF;
	x = (x | (x >> 8u)) & 0x0000FFFF;
	return (x & 0x0000FFFF);
}
uint lt_undilate_3(in uint x) {
	x = (x * 0x00015) & 0x0E070381;
	x = (x * 0x01041) & 0x0FF80001;
	x = (x * 0x40001) & 0x0FFC0000;
	return ((x >> 18u) & 0x0000FFFF);
}

/*
*/

/*
Encode node
===========
Dilate components and interleave.
Might overflow.
*********************************************/
void lt_encode_2_15(in uint level, in uvec2 p, out uint key) {
	key = level;                  // ---- ---- ---- ---- ---- ---- ---- llll
	key|= lt_dilate_2(p.x) << 4u; // -x-x -x-x -x-x -x-x -x-x -x-x -x-x llll
	key|= lt_dilate_2(p.y) << 5u; // yxyx yxyx yxyx yxyx yxyx yxyx yxyx llll
}
void lt_encode_2_30(in uint level, in uvec2 p, out uvec2 key) {
	uvec2 lsb = p & 0xFFFF;
	uvec2 msb = (p >> 16u) & 0x3FFF;
	key.x = level;                    // ---- ---- ---- ---- ---- ---- ---l llll
	key.x|= lt_dilate_2(msb.x) << 6u; // -x-x -x-x -x-x -x-x -x-x -x-x -x-l llll
	key.x|= lt_dilate_2(msb.y) << 7u; // yxyx yxyx yxyx yxyx yxyx yxyx yx-l llll
	key.y = lt_dilate_2(lsb.x);       // -x-x -x-x -x-x -x-x -x-x -x-x -x-x -x-x
	key.y|= lt_dilate_2(lsb.y) << 1u; // yxyx yxyx yxyx yxyx yxyx yxyx yxyx yxyx
}
void lt_encode_3_9(in uint level, in uvec3 p, out uint key) {
	key = level;                  // ---- ---- ---- ---- ---- ---- ---- llll
	key|= lt_dilate_3(p.x) << 4u; // ---- --x- -x-- x--x --x- -x-- x--x llll
	key|= lt_dilate_3(p.y) << 5u; // ---- -yx- yx-y x-yx -yx- yx-y x-yx llll
	key|= lt_dilate_3(p.z) << 6u; // ---- zyxz yxzy xzyx zyxz yxzy xzyx llll
}
void lt_encode_3_19(in uint level, in uvec3 p, out uvec2 key) {
	uvec3 lsb = p & 0xFFFF;
	uvec3 msb = (p >> 16u) & 0x3;
	key.x = level;                    // ---- ---- ---- ---- ---- ---- ---l llll
	key.x|= lt_dilate_3(msb.x) << 6u; // ---- -x-- x--x --x- -x-- x--x --xl llll
	key.x|= lt_dilate_3(msb.y) << 7u; // ---- yx-y x-yx -yx- yx-y x-yx -yxl llll
	key.x|= lt_dilate_3(msb.z) << 8u; // ---z yxzy xzyx zyxz yxzy xzyx zyxl llll
	key.y = lt_dilate_3(lsb.x);       // ---- x--x --x- -x-- x--x --x- -x-- x--x
	key.y|= lt_dilate_3(lsb.y) << 1u; // ---y x-yx -yx- yx-y x-yx -yx- yx-y x-yx
	key.y|= lt_dilate_3(lsb.z) << 2u; // --zy xzyx zyxz yxzy xzyx zyxz yxzy xzyx
}

/*
Decode node
===========
Undilate components and de-interleave.
*********************************************/
void lt_decode_2_15(in uint key, out uint level, out uvec2 p) {
	level = key & 0xF;
	p.x = lt_undilate_2((key >> 4u) & 0x05555555); // ---- -x-x -x-x -x-x -x-x -x-x -x-x -x-x
	p.y = lt_undilate_2((key >> 5u) & 0x05555555); // ---- -y-y --y- -y-- y--y --y- -y-- y--y
}
void lt_decode_2_30(in uvec2 key, out uint level, out uvec2 p) {
	level = key.x & 0x1F;
	p.x = lt_undilate_2(key.y & 0x55555555);
	p.y = lt_undilate_2((key.y >> 1u) & 0x55555555);
	p.x|= lt_undilate_2((key.x >> 6u) & 0x01555555) << 16u; // ---- ---x -x-x -x-x -x-x -x-x -x-x -x-x
	p.y|= lt_undilate_2((key.x >> 7u) & 0x01555555) << 16u; // ---- ---x -y-y -y-y -y-y -y-y -y-y -y-y
}
void lt_decode_3_9(in uint key, out uint level, out uvec3 p) {
	level = key & 0xF;
	p.x = lt_undilate_3((key >> 4u) & 0x01249249); // ---- ---x --x- -x-- x--x --x- -x-- x--x
	p.y = lt_undilate_3((key >> 5u) & 0x01249249); // ---- ---y --y- -y-- y--y --y- -y-- y--y
	p.z = lt_undilate_3((key >> 6u) & 0x01249249); // ---- ---z --z- -z-- z--z --z- -z-- z--z
}
void lt_decode_3_19(in uvec2 key, out uint level, out uvec3 p) {
	level = key.x & 0x1F;
	p.x = lt_undilate_3( key.y        & 0x09249249); // ---- x--x --x- -x-- x--x --x- -x-- x--x
	p.y = lt_undilate_3((key.y >> 1u) & 0x09249249); // ---- y--y --y- -y-- y--y --y- -y-- y--y
	p.z = lt_undilate_3((key.y >> 2u) & 0x09249249); // ---- z--z --z- -z-- z--z --z- -z-- z--z
	p.x|= lt_undilate_2((key.x >> 5u) & 0x00249249) << 16u; // ---- ---- --x- -x-- x--x --x- -x-- x--x
	p.y|= lt_undilate_2((key.x >> 6u) & 0x00249249) << 16u; // ---- ---- --y- -y-- y--y --y- -y-- y--y
	p.z|= lt_undilate_2((key.x >> 7u) & 0x00249249) << 16u; // ---- ---- --z- -z-- z--z --z- -z-- z--z
}

/*
get children nodes
==================
32bit 2tree:
   ---- ---- ---- ---- ---- ---- ---- 0000
-> ---- ---- ---- ---- ---- ---- --00 0001
-> ---- ---- ---- ---- ---- ---- --01 0001
-> ---- ---- ---- ---- ---- ---- --10 0001
-> ---- ---- ---- ---- ---- ---- --11 0001
==
32bit 3tree:
   ---- ---- ---- ---- ---- ---- ---- 0000
-> ---- ---- ---- ---- ---- ---- -000 0001
-> ---- ---- ---- ---- ---- ---- -001 0001
-> ---- ---- ---- ---- ---- ---- -010 0001
-> ---- ---- ---- ---- ---- ---- -011 0001
-> ---- ---- ---- ---- ---- ---- -100 0001
-> ---- ---- ---- ---- ---- ---- -101 0001
-> ---- ---- ---- ---- ---- ---- -110 0001
-> ---- ---- ---- ---- ---- ---- -111 0001
*********************************************/
void lt_children_2_15(in uint key, out uint children[4]) {
	key = (++key & 0xF) | ((key & ~0xF) << 2u);
	children[0] = key;        // ---- ---- ---- ---- ---- ---- --00 ----
	children[1] = key | 0x10; // ---- ---- ---- ---- ---- ---- --01 ----
	children[2] = key | 0x20; // ---- ---- ---- ---- ---- ---- --10 ----
	children[3] = key | 0x30; // ---- ---- ---- ---- ---- ---- --11 ----
}
void lt_children_2_30(in uvec2 key, out uvec2 children[4]) {
	children[0].x = (++key.x & 0x1F) | ((key.x & ~0x1F) << 2u);
	children[0].x|= (key.y >> 24u) & 0xC0; // save msb of msb part
	children[0].y = key.y << 2u;           // ---- ---- ---- ---- ---- ---- ---- --00
	children[1] = uvec2(children[0].x, children[0].y | 0x1); // ---- ---- ---- ---- ---- ---- ---- --01
	children[2] = uvec2(children[0].x, children[0].y | 0x2); // ---- ---- ---- ---- ---- ---- ---- --10
	children[3] = uvec2(children[0].x, children[0].y | 0x3); // ---- ---- ---- ---- ---- ---- ---- --11
}
void lt_children_3_9(in uint key, out uint children[8]) {
	// TODO
}
void lt_children_3_19(in uvec2 key, out uvec2 children[8]) {
	// TODO
}

/*
Get parent node
===============
32bit 2tree:
   ---- ---- ---- ---- ---- ---- --01 0001
-> ---- ---- ---- ---- ---- ---- ---- 0000
==
32bit 3tree:
   ---- ---- ---- ---- ---- ---- -001 0001
-> ---- ---- ---- ---- ---- ---- ---- 0000
==
64bit 2tree:
   ---- ---- ---- ---- ---- ---- ---0 0001 lsb
   ---- ---- ---- ---- ---- ---- ---- --01 msb
-> ---- ---- ---- ---- ---- ---- ---0 0000 lsb
-> ---- ---- ---- ---- ---- ---- ---- ---- msb
*********************************************/
uint lt_parent_2_15(in uint key) {
	return ((--key & 0xF) | ((key >> 2u) & 0x3FFFFFF0));
}
uvec2 lt_parent_2_30(in uvec2 key) {
	uvec2 r;
	r.x = (--key.x & 0x1F);
	r.x|= (key.x >> 2u) & 0x3FFFFFC0;
	r.y = (key.x << 24u) & 0xC0000000; // restore msb
	r.y|= (key.y >> 2u);
	return r;
}
uint lt_parent_3_9(in uint key) {
	return ((--key & 0xF) | ((key >> 3u) & 0x1FFFFFF0));
}
uvec2 lt_parent_3_9(in uvec2 key) {
	// TODO
	return uvec2(0);
}

/*
Is leaf
=======
Test level bits
*********************************************/
bool lt_is_leaf_2_15(in uint key) {
	return ((key & 0xF) == 0xE);
}
bool lt_is_leaf_2_30(in uvec2 key) {
	return ((key.x & 0x1F) == 0x1D);
}
bool lt_is_leaf_3_9(in uint key) {
	return ((key & 0xF) == 0x8);
}
bool lt_is_leaf_3_19(in uvec2 key) {
	// TODO
	return false;
}

/*
Is root
=======
Test level bits
*********************************************/
bool lt_is_root_2_15(in uint key) {
	return ((key & 0xF) == 0x0);
}
bool lt_is_root_2_30(in uvec2 key) {
	return ((key.x & 0x1F) == 0x00);
}
bool lt_is_root_3_9(in uint key) {
	return ((key & 0xF) == 0x0);
}
bool lt_is_root_3_19(in uvec2 key) {
	// TODO
	return false;
}

/*
Node orientation
================
Find the most significant bits in the morton code 
and test there value. Results are undefined for the root node.
*********************************************/
bool lt_is_left_2_15(in uint key) {
	return ((key & 0x10) == 0x00); // ---- ---- ---- ---- ---- ---- ---0 ----
}
bool lt_is_right_2_15(in uint key) {
	return ((key & 0x10) == 0x10); // ---- ---- ---- ---- ---- ---- ---1 ----
}
bool lt_is_upper_2_15(in uint key) {
	return ((key & 0x20) == 0x00); // ---- ---- ---- ---- ---- ---- --0- ----
}
bool lt_is_lower_2_15(in uint key) {
	return ((key & 0x20) == 0x20); // ---- ---- ---- ---- ---- ---- --1- ----
}

bool lt_is_left_2_30(in uvec2 key) {
	return ((key.y & 0x1) == 0x0); // ---- ---- ---- ---- ---- ---- ---- ---0
}
bool lt_is_right_2_30(in uvec2 key) {
	return ((key.y & 0x1) == 0x1); // ---- ---- ---- ---- ---- ---- ---- ---1
}
bool lt_is_upper_2_30(in uvec2 key) {
	return ((key.y & 0x2) == 0x0); // ---- ---- ---- ---- ---- ---- ---- --0-
}
bool lt_is_lower_2_30(in uvec2 key) {
	return ((key.y & 0x2) == 0x2); // ---- ---- ---- ---- ---- ---- ---- --1-
}

bool lt_is_left_3_9(in uint key) {
	return ((key & 0x10) == 0x00); // ---- ---- ---- ---- ---- ---- ---0 ----
}
bool lt_is_right_3_9(in uint key) {
	return ((key & 0x10) == 0x10); // ---- ---- ---- ---- ---- ---- ---1 ----
}
bool lt_is_upper_3_9(in uint key) {
	return ((key & 0x20) == 0x00); // ---- ---- ---- ---- ---- ---- --0- ----
}
bool lt_is_lower_3_9(in uint key) {
	return ((key & 0x20) == 0x20); // ---- ---- ---- ---- ---- ---- --1- ----
}
bool lt_is_front_3_9(in uint key) {
	return ((key & 0x40) == 0x00); // ---- ---- ---- ---- ---- ---- -0-- ----
}
bool lt_is_back_3_9(in uint key) {
	return ((key & 0x40) == 0x40); // ---- ---- ---- ---- ---- ---- -1-- ----
}

/*
Node orientation ctd
====================
Tells where the node was generated relatively to its parent.
Simply test the code's last bits.
*********************************************/
bool lt_is_upper_left_2_15(in uint key) {
	return ((key & 0x30) == 0x00); // ---- ---- ---- ---- ---- ---- --00 ----
}
bool lt_is_upper_right_2_15(in uint key) {
	return ((key & 0x30) == 0x10); // ---- ---- ---- ---- ---- ---- --01 ----
}
bool lt_is_lower_left_2_15(in uint key) {
	return ((key & 0x30) == 0x20); // ---- ---- ---- ---- ---- ---- --10 ----
}
bool lt_is_lower_right_2_15(in uint key) {
	return ((key & 0x30) == 0x30); // ---- ---- ---- ---- ---- ---- --11 ----
}

bool lt_is_upper_left_2_30(in uvec2 key) {
	return ((key.y & 0x3) == 0x0); // ---- ---- ---- ---- ---- ---- ---- --00
}
bool lt_is_upper_right_2_30(in uvec2 key) {
	return ((key.y & 0x3) == 0x1); // ---- ---- ---- ---- ---- ---- ---- --01
}
bool lt_is_lower_left_2_30(in uvec2 key) {
	return ((key.y & 0x3) == 0x2); // ---- ---- ---- ---- ---- ---- ---- --10
}
bool lt_is_lower_right_2_30(in uvec2 key) {
	return ((key.y & 0x3) == 0x3); // ---- ---- ---- ---- ---- ---- ---- --11
}

bool lt_is_front_upper_left_3_9(in uint key) {
	return ((key & 0x70) == 0x00); // ---- ---- ---- ---- ---- ---- -000 ----
}
bool lt_is_front_upper_right_3_9(in uint key) {
	return ((key & 0x70) == 0x10); // ---- ---- ---- ---- ---- ---- -001 ----
}
bool lt_is_front_lower_left_3_9(in uint key) {
	return ((key & 0x70) == 0x20); // ---- ---- ---- ---- ---- ---- -010 ----
}
bool lt_is_front_lower_right_3_9(in uint key) {
	return ((key & 0x70) == 0x30); // ---- ---- ---- ---- ---- ---- -011 ----
}
bool lt_is_back_upper_left_3_9(in uint key) {
	return ((key & 0x70) == 0x40); // ---- ---- ---- ---- ---- ---- -100 ----
}
bool lt_is_back_upper_right_3_9(in uint key) {
	return ((key & 0x70) == 0x50); // ---- ---- ---- ---- ---- ---- -101 ----
}
bool lt_is_back_lower_left_3_9(in uint key) {
	return ((key & 0x70) == 0x60); // ---- ---- ---- ---- ---- ---- -110 ----
}
bool lt_is_back_lower_right_3_9(in uint key) {
	return ((key & 0x70) == 0x70); // ---- ---- ---- ---- ---- ---- -111 ----
}

/*
*/

/*
Get normalized position + optionally parent (in [0,1])
======================================================
Extract the normalized coordinates from the grid index.
May provoke floatting point precision issues.
*********************************************/
void lt_cell_2_15(in uint key, out vec2 cell, out float cell_size) {
	uvec2 p;
	uint l;
	lt_decode_2_15(key, l, p); // get cell pos
	cell_size = 1.0 / float(1u << l); // normalized size
	cell = p * cell_size;
}
void lt_cell_2_15(in uint key, out vec2 cell, out float cell_size, out vec2 parent_cell) {
	uvec2 p;
	uint l;
	lt_decode_2_15(key, l, p); // get cell pos
	cell_size = 1.0 / float(1u << l); // normalized size
	cell = p * cell_size;
	parent_cell = cell - cell_size
	            * vec2(lt_is_right_2_15(key), lt_is_lower_2_15(key));
}

void lt_cell_2_30(in uvec2 key, out vec2 cell, out float cell_size) {
	uvec2 p;
	uint l;
	lt_decode_2_30(key, l, p); // get cell pos
	cell_size = 1.0 / float(1u << l); // normalized size
	cell = p * cell_size;
}
void lt_cell_2_30(in uvec2 key, out vec2 cell, out float cell_size, out vec2 parent_cell) {
	uvec2 p;
	uint l;
	lt_decode_2_30(key, l, p); // get cell pos
	cell_size = 1.0 / float(1u << l); // normalized size
	cell = p * cell_size;
	parent_cell = cell - cell_size
	            * vec2(lt_is_right_2_30(key), lt_is_lower_2_30(key));
}

void lt_cell_3_9(in uint key, out vec3 cell, out float cell_size) {
	uvec3 p;
	uint l;
	lt_decode_3_9(key, l, p); // get cell pos
	cell_size = 1.0 / float(1u << l); // normalized size
	cell = p * cell_size;
}
void lt_cell_3_9(in uint key, out vec3 cell, out float cell_size, out vec3 parent_cell) {
	uvec3 p;
	uint l;
	lt_decode_3_9(key, l, p); // get cell pos
	cell_size = 1.0 / float(1u << l); // normalized size
	cell = p * cell_size;
	parent_cell = cell - cell_size * vec3(lt_is_right_3_9(key),
	                                      lt_is_lower_3_9(key),
	                                      lt_is_upper_3_9(key));
}

#endif

