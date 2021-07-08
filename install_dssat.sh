#!/bin/bash
echo ""
echo "Installing dependencies"
echo ""
sudo apt-get install git
sudo apt-get install gfortran
sudo apt-get install cmake
echo ""
echo "Cloning DSSAT CSM"
echo ""
cd /tmp
git clone https://github.com/DSSAT/dssat-csm-os.git
echo ""
echo "Cloning DSSAT DATA"
echo ""
git clone https://github.com/DSSAT/dssat-csm-data.git
echo ""
echo "Compiling DSSAT at ~/dssat"
echo ""
cd dssat-csm-os
mkdir BUILD
cd BUILD
cmake -DCMAKE_INSTALL_PREFIX=~/dssat ..
# sudo make -j && sudo make install
sudo make && sudo make install
cd /tmp
sudo chmod 757 -R   ~/dssat
sudo cp -r dssat-csm-data/* ~/dssat
echo ""
echo "DSSAT files added"
echo ""
rm -rf dssat-csm-data
rm -rf dssat-csm-os
echo ""
echo "END OF DSSAT INSTALLATION"
echo ""

