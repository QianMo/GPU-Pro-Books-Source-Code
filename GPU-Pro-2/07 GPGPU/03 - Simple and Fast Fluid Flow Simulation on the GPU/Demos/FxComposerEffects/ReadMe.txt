In order of importance:

///< 1.
Make SURE to open fxcomposer and set the desired directX version 9 BEFORE loading the project file as it seems to bug when changing dx versions on the fly and mostly when using the dx10 version.

///< 2.  To animate:
-Press the "Play" button located at the top right of your screen.
-Select the Solver Material in the Assets panel.
-Set the Initialize Domain value to true in the Properties (normaly to the right of your screen).
-Set the Initialize Domain value to false.

///< Juste a note:
We use the fact that it is possible to read&write to the same texture in directx9 but it is not good pratice.
We left out the dx10 version as point 1. was a big issue for first time users and also note that texture copies are required with later version then 9. Can
`t read and write to the same texture :)

Martin.