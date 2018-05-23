REQUIREMENTS
====================================================================

install latest DirectX Runtime
http://www.microsoft.com/en-us/download/confirmation.aspx?id=8109

install visual studio 2010 c++ x86/x64 redist
http://www.microsoft.com/de-de/download/confirmation.aspx?id=5555


CONFIGURATION
====================================================================

You can change the atlas resolution by setting the corresponding value in the Settings.xml.


KNOWN ISSUES
====================================================================

We encountered problems on multi-GPU systems.
If you have more than one GPU try to deactivate additional devices in the Windows Device Manager.

Since we are using DirectX11 features make sure your system supports D3D_FEATURE_LEVEL_11_0.