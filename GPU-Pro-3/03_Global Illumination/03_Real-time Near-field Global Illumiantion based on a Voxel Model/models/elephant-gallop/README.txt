
I have made available all mesh data from my paper:

  Robert W. Sumner, Jovan Popovic. Deformation Transfer for Triangle Meshes.
  ACM Transactions on Graphics. 23, 3. August 2004
  http://graphics.csail.mit.edu/~sumner/research/deftransfer/

as well as the elephant example that I showed during my presentation
at SIGGRAPH 2004.  To access this data, please visit:

  http://graphics.csail.mit.edu/~sumner/research/deftransfer/data.html

If you use these meshes in a manner that relates to my paper, please
reference it.

For more information, questions, comments, problems, or suggestions,
please email Bob Sumner at sumner@graphics.csail.mit.edu.

The data is grouped into these directories:

horse-poses     | Horse poses from Figure 1.
horse-gallop    | Horse gallop animation from the video.
horse-collapse  | Horse collapse animation from Figure 6 and the video.
camel-poses     | Camel poses from Figure 1.
camel-gallop    | Camel gallop animation from the video.
camel-collapse  | Camel collapse animation from Figure 6 and the video.
cat-poses       | Cat poses from Figure 5.
lion-poses      | Lion poses from Figure 5.
face-poses      | Face mask expressions from Figure 7.
head-poses      | Head expressions from Figure 7.
flamingo-poses  | Flamingo poses from Figure 8.
elephant-poses  | Elephant poses shown at SIGGRAPH 2004.
elephant-gallop | Elephant gallop animation, shown at SIGGRAPH 2004.

Every mesh is triangulated and in .obj format.  This is an ASCII text,
line-based format.  Comment lines begin with "#", vertices with "v",
vertex normals with "vn", and triangles with "f".  The triangle lines
contain indices into the vertex and vertex normal lists.  These
indices are one-based (ie, starting with one -- not with zero).  No
texture coordinates are included in any of the meshes.

Each directory contains one mesh with "-reference" in the filename.
This means that it was the reference mesh for that particular example,
as indicated in the figures from the paper.  The horse, cat, and face
were used as source meshes.  The camel, lion, head, flamingo, and
elephant were target meshes.  Thus, the poses in the target mesh
directories were created by deforming the reference mesh according to
the techniques described in the paper.

The horse, cat, reference lion, and reference head meshes are
originally exported from the software "Poser" by Curious Labs
(www.curiouslabs.com).  The reference camel, reference flamingo, and
reference elephant were originally from the De Espona model library
(www.deespona.com).  I purchased the horse gallop animation (not the
mesh -- just the animation) from DAZ Productions (daz3d.com).  All
meshes were processed slightly by me (eg, triangulating, removing
extra connected components, etc.).  The face meshes were captured with
a facial scanning system.  For more information, see the SIGGRAPH 2004
technical sketch:

  Daniel Vlasic, Matt Brand, Hanspeter Pfister, Jovan Popovic.
  Multilinear Models for Facial Synthesis.

A few caveats about these meshes:

  - Several of the meshes -- especially those exported from Poser --
    contain small "errors" that make them non-manifold and
    non-watertight.  For example, the horse has a small slit that is
    not closed one of its hoofs.  These "errors" were present in the
    exported model.

  - The camel's ears are pierced, making it a genus three mesh.

  - In some poses, the cat's tail is unnaturally twisted a full 180
    degrees.  This twist was included in the Poser sample poses for
    the cat.  Since the cat poses were transferred onto the lion, the
    lion's tail dutifully twists in the same way.

  - I made the collapsing horse animation using Maya cloth and a lot
    of parameter tweaking.
