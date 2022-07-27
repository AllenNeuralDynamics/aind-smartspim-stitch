@echo off

mkdir "%2"
mkdir "%2\xmls"
echo "Created directory in path: '%2'"

echo "Import step"
:: Step 1
terastitcher --import --volin="%1" --ref1=1 --ref2=-2 --ref3=3 --vxl1=0.8 --vxl2=0.8 --vxl3=1 --projout="%2\xmls\xml_import.xml" --sparse_data

echo "Align step"
:: Step 2
mpiexec -n 16 python ./pyscripts/parastitcher.py -2 --projin="%2\xmls\xml_import.xml" --projout="%2\xmls\xml_displcomp_par2.xml" --subvoldim=100 > "%2\xmls\step2par.txt"

echo "Projection step"
:: Step 3
terastitcher --displproj --projin="%2\xmls\xml_displcomp_par2.xml" --projout="%2\xmls\xml_displproj.xml"

echo "Threshold step"
:: Step 4
terastitcher --displthres --projin="%2\xmls\xml_displproj.xml" --projout="%2\xmls\xml_displthres.xml" --threshold=0.7

echo "Placing tiles step"
:: Step 5
terastitcher --placetiles --projin="%2\xmls\xml_displthres.xml" --projout="%2\xmls\xml_merging.xml"

echo "Merge step"
:: Step 6
mpiexec -n 9 python ./pyscripts/parastitcher.py -6 --projin="%2\xmls\xml_merging.xml" --volout="%2" --resolutions=0 --slicewidth=2000 --sliceheight=2000

:: Generate 3D tiff stacks
:: mpiexec -n 9 python ./pyscripts/parastitcher.py -6 --projin=..\TestData\mouse.cerebellum.300511.sub3\tomo300511_subv3\xml_merging.xml --volout=..\TestData\merge_step_3d --resolutions=0 --slicewidth=2000 --sliceheight=2000 --slicedepth=100 --volout_plugin="TiledXY|3Dseries" > ..\TestData\merge_step_3d\step6par_3dseries.txt 