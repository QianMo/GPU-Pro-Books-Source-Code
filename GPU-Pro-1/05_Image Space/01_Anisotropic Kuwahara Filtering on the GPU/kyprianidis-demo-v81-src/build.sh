#!/bin/sh
rm -rf jkyprian-akf.app jkyprian-akf.dmg
qmake -config release
make
macdeployqt jkyprian-akf.app -dmg
mv jkyprian-akf.dmg kyprianidis-demo-mac.dmg
