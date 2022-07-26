:: Step 1
terastitcher --import --volin=..\TestData\mouse.cerebellum.300511.sub3\tomo300511_subv3 --ref1=1 --ref2=-2 --ref3=3 --vxl1=0.8 --vxl2=0.8 --vxl3=1 --projout=..\TestData\mouse.cerebellum.300511.sub3\tomo300511_subv3\xml_import.xml

:: Step 2
mpiexec -n 9 python ./pyscripts/parastitcher.py -2 --projin=..\TestData\mouse.cerebellum.300511.sub3\tomo300511_subv3\xml_import.xml --projout=..\TestData\mouse.cerebellum.300511.sub3\tomo300511_subv3\xml_displcomp_par2.xml --subvoldim=100 > ..\TestData\mouse.cerebellum.300511.sub3\tomo300511_subv3\step2par.txt

:: Step 3
terastitcher --displproj --projin=..\TestData\mouse.cerebellum.300511.sub3\tomo300511_subv3\xml_displcomp_par2.xml --projout=..\TestData\mouse.cerebellum.300511.sub3\tomo300511_subv3\xml_displproj.xml

:: Step 4
terastitcher --displthres --projin=..\TestData\mouse.cerebellum.300511.sub3\tomo300511_subv3\xml_displproj.xml --projout=..\TestData\mouse.cerebellum.300511.sub3\tomo300511_subv3\xml_displthres.xml --threshold=0.7

:: Step 5
terastitcher --placetiles --projin=..\TestData\mouse.cerebellum.300511.sub3\tomo300511_subv3\xml_displthres.xml --projout=..\TestData\mouse.cerebellum.300511.sub3\tomo300511_subv3\xml_merging.xml

:: Step 6
:: mpiexec -n 9 python ./pyscripts/parastitcher.py -6 --projin=..\TestData\mouse.cerebellum.300511.sub3\tomo300511_subv3\xml_merging.xml --volout=..\TestData\merge_step_results_par --resolutions=012345 > ..\TestData\mouse.cerebellum.300511.sub3\tomo300511_subv3\step6par.txt

mpiexec -n 9 python ./pyscripts/parastitcher.py -6 --projin=..\TestData\mouse.cerebellum.300511.sub3\tomo300511_subv3\xml_merging.xml --volout=..\TestData\merge_step_results_par --resolutions=0 --slicewidth=2000 --sliceheight=2000 > ..\TestData\mouse.cerebellum.300511.sub3\tomo300511_subv3\step6par.txt

:: Generate 3D tiff stacks
:: mpiexec -n 9 python ./pyscripts/parastitcher.py -6 --projin=..\TestData\mouse.cerebellum.300511.sub3\tomo300511_subv3\xml_merging.xml --volout=..\TestData\merge_step_3d --resolutions=0 --slicewidth=2000 --sliceheight=2000 --slicedepth=100 --volout_plugin="TiledXY|3Dseries" > ..\TestData\merge_step_3d\step6par_3dseries.txt 