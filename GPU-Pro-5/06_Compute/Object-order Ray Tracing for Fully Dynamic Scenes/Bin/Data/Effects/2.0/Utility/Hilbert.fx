#ifndef BE_UTILITY_HILBERT_H
#define BE_UTILITY_HILBERT_H

#define BE_UTILITY_HILBERT_DIM 2
#include "Utility/Hilbert.fx"
#undef BE_UTILITY_HILBERT_DIM

#define BE_UTILITY_HILBERT_DIM 3
#include "Utility/Hilbert.fx"
#undef BE_UTILITY_HILBERT_DIM

#define BE_UTILITY_HILBERT_DIM 4
#include "Utility/Hilbert.fx"
#undef BE_UTILITY_HILBERT_DIM

#define BE_UTILITY_HILBERT_DIM 5
#include "Utility/Hilbert.fx"
#undef BE_UTILITY_HILBERT_DIM

#endif

#ifdef BE_UTILITY_HILBERT_DIM

/*
This code assumes the following:
The macro ORDER corresponds to the order of curve and is 32,
thus coordinates of points are 32 bit values.
A uint should be a 32 bit unsigned integer.
The macro DIM corresponds to the number of dimensions in a
space.
The derived-key of a HCODE is stored in an HCODE which is an
array of uint. The bottom bit of a derived-key is held in the
bottom bit of the hcode[0] element of an HCODE and the top bit
of a derived-key is held in the top bit of the hcode[DIM-1]
element of and HCODE.
g_mask is a global array of masks which helps simplyfy some
calculations - it has DIM elements. In each element, only one
bit is zeo valued - the top bit in element no. 0 and the bottom
bit in element no. (DIM - 1). eg.
#if DIM == 5 const uint g_mask[] = {16, 8, 4, 2, 1}; #endif
#if DIM == 6 const uint g_mask[] = {32, 16, 8, 4, 2, 1}; #endif
etc...
*/

#define ORDER 32
#define DIM BE_UTILITY_HILBERT_DIM
#define DEC1(i, d) hilbert_ ## i ## d
#define DEC(i) DEC1(i, DIM)
#define HCODE DEC(code)

struct HCODE
{
	uint hcode[DIM];
};

uint DEC(mask)(uint i)
{
	return (uint) 1 << ((uint) (DIM - 1) - i);
}

/*===========================================================*/
/* calc_P */
/*===========================================================*/
uint DEC(calc_P)(int i, HCODE H)
{
	int element;
	uint P, temp1, temp2;
	element = i / ORDER;
	P = H.hcode[element];
	if (i % ORDER > ORDER - DIM)
	{
		temp1 = H.hcode[element + 1];
		P >>= i % ORDER;
		temp1 <<= ORDER - i % ORDER;
		P |= temp1;
	}
	else
		P >>= i % ORDER;
	/* P is a DIM bit hcode */
	/* the & masks out spurious highbit values */
#if DIM < ORDER
	P &= (1 << DIM) -1;
#endif
	return P;
}

/*===========================================================*/
/* calc_P2 */
/*===========================================================*/
uint DEC(calc_P2)(uint S)
{
	int i;
	uint P;
	P = S & DEC(mask)(0);
	for (i = 1; i < DIM; i++)
		if( S & DEC(mask)(i) ^ (P >> 1) & DEC(mask)(i))
			P |= DEC(mask)(i);
	return P;
}

/*===========================================================*/
/* calc_J */
/*===========================================================*/
uint DEC(calc_J)(uint P)
{
	int i;
	uint J;
	J = DIM;
	for (i = 1; i < DIM; i++)
		if ((P >> i & 1) == (P & 1))
			continue;
		else
			break;
	if (i != DIM)
		J -= i;
	return J;
}

/*===========================================================*/
/* calc_T */
/*===========================================================*/
uint DEC(calc_T)(uint P)
{
	if (P < 3)
		return 0;
	if (P % 2)
		return (P - 1) ^ (P - 1) / 2;
	return (P - 2) ^ (P - 2) / 2;
}


/*===========================================================*/
/* calc_tS_tT */
/*===========================================================*/
uint DEC(calc_tS_tT)(uint xJ, uint val)
{
	uint retval, temp1, temp2;
	retval = val;
	if (xJ % DIM != 0)
	{
		temp1 = val >> xJ % DIM;
		temp2 = val << DIM - xJ % DIM;
		retval = temp1 | temp2;
		retval &= ((uint)1 << DIM) - 1;
	}

	return retval;
}

/*===========================================================*/
/* H_decode */
/*===========================================================*/
/* For mapping from one dimension to DIM dimensions */
HCODE DEC(decode)(HCODE H)
{
	uint mask = (uint)1 << ORDER - 1;
	uint A, W = 0, S, tS, T, tT, J, P = 0, xJ;
	HCODE pt = (HCODE) 0;
	int i = ORDER * DIM - DIM, j;
	P = DEC(calc_P)(i, H);
	J = DEC(calc_J)(P);
	xJ = J - 1;
	A = S = tS = P ^ P / 2;
	T = DEC(calc_T)(P);
	tT = T;
	/*--- distrib bits to coords ---*/
	for (j = DIM - 1; P > 0; P >>=1, j--)
		if (P & 1)
			pt.hcode[j] |= mask;
	for (i -= DIM, mask >>= 1; i >=0; i -= DIM, mask >>= 1)
	{
		P = DEC(calc_P)(i, H);
		S = P ^ P / 2;
		tS = DEC(calc_tS_tT)(xJ, S);
		W ^= tT;
		A = W ^ tS;
		/*--- distrib bits to coords ---*/
		for (j = DIM - 1; A > 0; A >>=1, j--)
			if (A & 1)
				pt.hcode[j] |= mask;
		if (i > 0)
		{
			T = DEC(calc_T)(P);
			tT = DEC(calc_tS_tT)(xJ, T);
			J = DEC(calc_J)(P);
			xJ += J - 1;
		}
	}
	return pt;
}

/*===========================================================*/
/* H_encode */
/*===========================================================*/
/* For mapping from DIM dimensions to one dimension */
HCODE DEC(encode)(HCODE pt)
{
	uint mask = (uint)1 << ORDER - 1;
	uint element, A, W = 0, S, tS, T, tT, J, P = 0, xJ;
	HCODE h = (HCODE) 0;
	int i = ORDER * DIM - DIM, j;
	for (j = A = 0; j < DIM; j++)
		if (pt.hcode[j] & mask)
			A |= DEC(mask)(j);
	S = tS = A;
	P = DEC(calc_P2)(S);
	/* add in DIM bits to hcode */
	element = i / ORDER;
	if (i % ORDER > ORDER - DIM)
	{
		h.hcode[element] |= P << i % ORDER;
		h.hcode[element + 1] |= P >> ORDER - i % ORDER;
	}
	else
		h.hcode[element] |= P << i - element * ORDER;
	J = DEC(calc_J)(P);
	xJ = J - 1;
	T = DEC(calc_T)(P);
	tT = T;
	for (i -= DIM, mask >>= 1; i >=0; i -= DIM, mask >>= 1)
	{
		for (j = A = 0; j < DIM; j++)
			if (pt.hcode[j] & mask)
				A |= DEC(mask)(j);
		W ^= tT;
		tS = A ^ W;
		S = DEC(calc_tS_tT)(xJ, tS);
		P = DEC(calc_P2)(S);
		/* add in DIM bits to hcode */
		element = i / ORDER;
		if (i % ORDER > ORDER - DIM)
		{
			h.hcode[element] |= P << i % ORDER;
			h.hcode[element + 1] |= P >> ORDER - i % ORDER;
		}
		else
			h.hcode[element] |= P << i - element * ORDER;
		if (i > 0)
		{
			T = DEC(calc_T)(P);
			tT = DEC(calc_tS_tT)(xJ, T);
			J = DEC(calc_J)(P);
			xJ += J - 1;
		}
	}
	return h;
}

#undef ORDER
#undef DIM
#undef DEC1
#undef DEC
#undef HCODE

#endif