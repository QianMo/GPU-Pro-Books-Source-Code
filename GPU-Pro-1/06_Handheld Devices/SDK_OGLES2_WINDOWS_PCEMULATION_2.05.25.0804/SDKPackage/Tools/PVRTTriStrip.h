/******************************************************************************

 @File         PVRTTriStrip.h

 @Title        PVRTTriStrip

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  Strips a triangle list.

******************************************************************************/
#ifndef _PVRTTRISTRIP_H_
#define _PVRTTRISTRIP_H_


/****************************************************************************
** Declarations
****************************************************************************/

/*!***************************************************************************
 @Function			PVRTTriStrip
 @Output			ppwStrips
 @Output			ppnStripLen
 @Output			pnStripCnt
 @Input				pwTriList
 @Input				nTriCnt
 @Description		Reads a triangle list and generates an optimised triangle strip.
*****************************************************************************/
void PVRTTriStrip(
	unsigned short			**ppwStrips,
	unsigned int			**ppnStripLen,
	unsigned int			*pnStripCnt,
	const unsigned short	* const pwTriList,
	const unsigned int		nTriCnt);


/*!***************************************************************************
 @Function			PVRTTriStripList
 @Modified			pwTriList
 @Input				nTriCnt
 @Description		Reads a triangle list and generates an optimised triangle strip. Result is
 					converted back to a triangle list.
*****************************************************************************/
void PVRTTriStripList(unsigned short * const pwTriList, const unsigned int nTriCnt);


#endif /* _PVRTTRISTRIP_H_ */

/*****************************************************************************
 End of file (PVRTTriStrip.h)
*****************************************************************************/
