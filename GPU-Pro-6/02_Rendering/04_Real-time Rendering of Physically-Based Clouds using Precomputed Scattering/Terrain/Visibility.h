
struct SBoundingBox
{
    float fMinX, fMaxX, fMinY, fMaxY, fMinZ, fMaxZ;
};

// Structure describing a plane
struct SPlane3D
{
    D3DXVECTOR3 Normal;
    float Distance;     //Distance from the coordinate system origin to the plane along normal direction
};

#pragma pack(1)
struct SViewFrustum
{
    SPlane3D LeftPlane, RightPlane, BottomPlane, TopPlane, NearPlane, FarPlane;
};
#pragma pack()
static UINT TEST_LEFT_PLANE   = (1 << 0);
static UINT TEST_RIGHT_PLANE  = (1 << 1);
static UINT TEST_BOTTOM_PLANE = (1 << 2);
static UINT TEST_TOP_PLANE    = (1 << 3);
static UINT TEST_NEAR_PLANE   = (1 << 4);
static UINT TEST_FAR_PLANE    = (1 << 5);
static UINT TEST_ALL_PLANES = (TEST_LEFT_PLANE | TEST_RIGHT_PLANE | TEST_BOTTOM_PLANE | TEST_TOP_PLANE | TEST_NEAR_PLANE | TEST_FAR_PLANE);

// Tests if bounding box is visible by the camera
static inline bool IsBoxVisible(const SViewFrustum &ViewFrustum, const SBoundingBox &Box, UINT uiPlaneFlags = TEST_ALL_PLANES)
{
    SPlane3D *pPlanes = (SPlane3D *)&ViewFrustum;
    // If bounding box is "behind" some plane, then it is invisible
    // Otherwise it is treated as visible
    for(int iViewFrustumPlane = 0; iViewFrustumPlane < 6; iViewFrustumPlane++)
    {
        if( !(uiPlaneFlags & (1<<iViewFrustumPlane)) )
            continue;

        SPlane3D *pCurrPlane = pPlanes + iViewFrustumPlane;
        D3DXVECTOR3 *pCurrNormal = &pCurrPlane->Normal;
        D3DXVECTOR3 MaxPoint;
        
        MaxPoint.x = (pCurrNormal->x > 0) ? Box.fMaxX : Box.fMinX;
        MaxPoint.y = (pCurrNormal->y > 0) ? Box.fMaxY : Box.fMinY;
        MaxPoint.z = (pCurrNormal->z > 0) ? Box.fMaxZ : Box.fMinZ;
        
        float DMax = D3DXVec3Dot( &MaxPoint, pCurrNormal ) + pCurrPlane->Distance;

        if( DMax < 0 )
            return false;
    }

    return true;
}



// Extract view frustum planes from the world-view-projection matrix
static inline void ExtractViewFrustumPlanesFromMatrix(const D3DXMATRIX &Matrix, SViewFrustum &ViewFrustum)
{
    // For more details, see Gribb G., Hartmann K., "Fast Extraction of Viewing Frustum Planes from the 
    // World-View-Projection Matrix" (the paper is available at 
    // http://www2.ravensoft.com/users/ggribb/plane%20extraction.pdf)

	// Left clipping plane 
    ViewFrustum.LeftPlane.Normal.x = Matrix._14 + Matrix._11; 
	ViewFrustum.LeftPlane.Normal.y = Matrix._24 + Matrix._21; 
	ViewFrustum.LeftPlane.Normal.z = Matrix._34 + Matrix._31; 
	ViewFrustum.LeftPlane.Distance = Matrix._44 + Matrix._41;

	// Right clipping plane 
	ViewFrustum.RightPlane.Normal.x = Matrix._14 - Matrix._11; 
	ViewFrustum.RightPlane.Normal.y = Matrix._24 - Matrix._21; 
	ViewFrustum.RightPlane.Normal.z = Matrix._34 - Matrix._31; 
	ViewFrustum.RightPlane.Distance = Matrix._44 - Matrix._41;

	// Top clipping plane 
	ViewFrustum.TopPlane.Normal.x = Matrix._14 - Matrix._12; 
	ViewFrustum.TopPlane.Normal.y = Matrix._24 - Matrix._22; 
	ViewFrustum.TopPlane.Normal.z = Matrix._34 - Matrix._32; 
	ViewFrustum.TopPlane.Distance = Matrix._44 - Matrix._42;

	// Bottom clipping plane 
	ViewFrustum.BottomPlane.Normal.x = Matrix._14 + Matrix._12; 
	ViewFrustum.BottomPlane.Normal.y = Matrix._24 + Matrix._22; 
	ViewFrustum.BottomPlane.Normal.z = Matrix._34 + Matrix._32; 
	ViewFrustum.BottomPlane.Distance = Matrix._44 + Matrix._42;

	// Near clipping plane 
	ViewFrustum.NearPlane.Normal.x = Matrix._13; 
	ViewFrustum.NearPlane.Normal.y = Matrix._23; 
	ViewFrustum.NearPlane.Normal.z = Matrix._33; 
	ViewFrustum.NearPlane.Distance = Matrix._43;

	// Far clipping plane 
	ViewFrustum.FarPlane.Normal.x = Matrix._14 - Matrix._13; 
	ViewFrustum.FarPlane.Normal.y = Matrix._24 - Matrix._23; 
	ViewFrustum.FarPlane.Normal.z = Matrix._34 - Matrix._33; 
	ViewFrustum.FarPlane.Distance = Matrix._44 - Matrix._43; 
}

