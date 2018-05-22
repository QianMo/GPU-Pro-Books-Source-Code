// -*- c++ -*-
/**
  \file SS_BumpMap.glsl
  \author Morgan McGuire, http://graphics.cs.williams.edu

  For use with G3D::SuperShader.
  This file is included into NonShadowedPass.pix and ShadowMappedLightPass.pix.

  \created 2007-12-18
  \edited  2011-06-17

  This file defines three variations of the following function:

  void bumpMap(in sampler2D normalBumpMap, in float bumpMapScale, in float bumpMapBias, in vec2 texCoord, in vec3 tan_X, in vec3 tan_Y, in vec3 tan_Z, in float backside, in vec3 tsE, out vec3 wsN, out vec2 offsetTexCoord);

  Input:
  normalBumpMap  .xyz = tangent-space shading normal, .a - 0.5 = displacement from geometric surface
  bumpMapScale   Multiplies the net displacement
  bumpMapBias    Added to normalBumpMap.a
  texCoord       texture coordinate in normalBumpMap
  tan_X          unit tangent-space X vector (usually, the texture-space "u" axis) in world space.  a.k.a. "tangent"
  tan_Y          unit tangent-space Y vector (usually, the texture-space "v" axis) in world space.  a.k.a. "binormal"
  tan_Z          unit tangent-space Z vector (usually the surface normal)
  backside       1.0 if this is is a front face, -1.0 if this is a backface with two-sided lighting
  tsE            unit vector to the center of projection, in tangent space

  Output:
  wsN            world-space shading normal at offsetTexCoord
  offsetTexCoord shading texture coordinate

  \sa G3D::BumpMap, G3D::UniversalMaterial
 */

/**
  Normal mapping
  Following the algorithm of Blinn '78
  */
void bumpMapBlinn78(in sampler2D normalBumpMap, in float bumpMapScale, in float bumpMapBias, in vec2 texCoord, in vec3 tan_X, in vec3 tan_Y, in vec3 tan_Z, in float backside, in vec3 tsE, out vec3 wsN, out vec2 offsetTexCoord, out vec3 tsN) {
        
    offsetTexCoord = texCoord;
            
    // Take the normal map values back to (-1, 1) range to compute a tangent space normal
    tsN = texture2D(normalBumpMap, offsetTexCoord).xyz * 2.0 + vec3(-1.0, -1.0, -1.0);

    // note that the columns might be slightly non-orthogonal due to interpolation
    mat3 tangentToWorld = mat3(tan_X, tan_Y, tan_Z);

    // Take the normal to world space
    wsN = (tangentToWorld * tsN) * backside;

    // For very tiny geometry the tangent space can become degenerate and produce a bad normal
    wsN = normalize((dot(wsN, wsN) > 0.001) ? wsN : tan_Z);
}


/**
    Parallax mapping
    Following the algorithm of Welsh '04
    */
void bumpMapWelsh04(in sampler2D normalBumpMap, in float bumpMapScale, in float bumpMapBias, in vec2 texCoord, in vec3 tan_X, in vec3 tan_Y, in vec3 tan_Z, in float backside, in vec3 tsE, out vec3 wsN, out vec2 offsetTexCoord, out vec3 tsN) {
    // Convert bumps to a world space distance
    float bump = (texture2D(normalBumpMap, texCoord).w - 0.5 + bumpMapBias) * bumpMapScale;
    
    // Offset the texture coord.  Note that texture coordinates are inverted (in the y direction)
    // from TBN space, so we must flip the y-axis.

    offsetTexCoord = texCoord.xy + vec2(tsE.x, -tsE.y) * bump;
            
    // Take the normal map values back to (-1, 1) range to compute a tangent space normal
    tsN = texture2D(normalBumpMap, offsetTexCoord).xyz * 2.0 + vec3(-1.0, -1.0, -1.0);

    // note that the columns might be slightly not orthogonal due to interpolation
    mat3 tangentToWorld = mat3(tan_X, tan_Y, tan_Z);

    // Take the normal to world space
    wsN = (tangentToWorld * tsN) * backside;

    // For very tiny geometry the tangent space can become degenerate and produce a bad normal
    wsN = normalize((dot(wsN, wsN) > 0.001) ? wsN : tan_Z);
}


/**
    Parallax Occlusion Mapping (POM)

    Following the algorithm of Tatarchuk I3D '06
     
    Linear search and linear interpolation after the hit.
    Constants have been adjusted to give high performance on GeForce cards.
    */
void bumpMapTatarchuk06(in sampler2D normalBumpMap, in float bumpMapScale, in float bumpMapBias, in vec2 texCoord, in vec3 tan_X, in vec3 tan_Y, in vec3 tan_Z, in float backside, in vec3 tsE, out vec3 wsN, out vec2 offsetTexCoord, out vec3 tsN) {
    // Actual texture coordinate that we'll use based on
    // parallax offset.  The z coordinate is the height of the
    // bump above the surface
    vec3 offsetCoord;

    // tsN is the Packed tangent space normals (to be unpacked below)
    

    // Back up the intersection since we should be entering the surface
    // at the top of the bounding box (bump = 1) and the initial intersection point 
    // is with bump = 0.5
    //
    // Note that we negate tsE because tsE points towards the eye
    // and we're walking away from the eye; also note that we negate
    // the y coordinate because our texture coords are inverted in y.
    offsetCoord.xy = texCoord - bumpMapBias * bumpMapScale * vec2(-tsE.x, tsE.y) / tsE.z;

    // Normalized height of the eye ray above the surface (i.e., not multiplied 
    // by bumpScale to avoid the cost of bump height on the inner loop.)
    offsetCoord.z = 1.0;
    const float MIN_STEPS = 2.0;

    // Increase the number of steps taken as the eye vector begins 
    // to point more horizontally (ala Tatarchuk).  Expand the range of tsE.z; it rarely gets near zero on its own.

    float numSteps = mix(max(float(PARALLAXSTEPS) * bumpMapScale * 2.0, MIN_STEPS), MIN_STEPS,  max(0.0, (tsE.z - 0.4) / 0.6) );
    vec3 tsStep;

    // Distance that we'll step in z (normal direction) for each 
    // iteration; by definition we have to cover bumpScale distance
    // in numSteps.
    tsStep.z = -1.0 / numSteps; 

    // Corresponding (x, y) step, by similar triangles; note that
    // tsStep.z is negative so we've negated the eye vector again,
    // giving us back the positive version.
    tsStep.xy = (vec2(tsE.x, -tsE.y) / tsE.z) * (bumpMapScale * tsStep.z);

    // Surface normal and bump map values.
    vec4 NB = texture2D(normalBumpMap, offsetCoord.xy);

    // March forward until the view ray sinks below the current
    // surface elevation.  At each point, offsetCoord will be the
    // next location, and we'll overshoot slightly.
    while (NB.w < offsetCoord.z) {
        offsetCoord += tsStep;
        NB = texture2D(normalBumpMap, offsetCoord.xy);
    }

    // We've overshot slightly.  Assume that the surface is piecewise linear ala Policarop and Oliveira '06, Tatarchuk '06
    //
    // By similar triangles, the overshoot is thus governed by the ratios:
    // 
    //    last ray elevation - last surface elevation             surface dist from last to true intersection             P
    //  -----------------------------------------------      =   -------------------------------------------------   = -------
    //  current surface elevation - current ray elevation         surface dist from current to true intersection          Q
    //
    // In other words, we want to step back by (tsStep * Q / (P + Q)).  The denominator of that ratio
    // can never be zero or we would be travelling parallel to the surface and should have hit on the
    // previous iteration.  However, it can be very small--so small that it underflows.
    //
    // P = (offsetCoord.z - tsStep.z) - prevNB.w;
    // Q = NB.w - offsetCoord.z;
    //
    // alpha = 
    // Q / (P + Q) = (NB.w - offsetCoord.z) / ( (offsetCoord.z - tsStep.z) - prevNB.w + NB.w - offsetCoord.z)
    //             = (NB.w - offsetCoord.z) / (NB.w -tsStep.z - prevNB.w)
        
    // (We could have tracked prevNB in the inner while loop, but in testing on NVIDIA GeForce 7800, 
    // it was faster to just perform one very-coherent texture read at the end than add instructions to
    // the inner loop)
    vec4 prevNB = texture2D(normalBumpMap, offsetCoord.xy - tsStep.xy);

    float denom = NB.w - tsStep.z - prevNB.w;
    denom = max(abs(denom), 0.125) * sign(denom);
    float alpha = (NB.w - offsetCoord.z) / denom;
    offsetCoord -= tsStep * alpha;

    // Take the tangent back appropriately
    tsN = mix(NB.xyz, prevNB.xyz, alpha) * 2.0 + vec3(-1.0, -1.0, -1.0);

    offsetTexCoord = offsetCoord.xy;

    // Note that the columns might be slightly not orthogonal due to interpolation
    mat3 tangentToWorld = mat3(tan_X, tan_Y, tan_Z * backside);

    // Take the normal to world space
    wsN = tangentToWorld * tsN;

    // For very tiny geometry the tangent space can become degenerate and produce a bad normal
    wsN = normalize((dot(wsN, wsN) > 0.001) ? wsN : tan_Z);
}

/* End SS_BumpMap.glsl */
