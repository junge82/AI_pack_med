# build c++ executable
1. mkdir build; cd build
2. cmake ..
3. make

# use c++ executable
- ./SP <process_type> <input_path> <output_path> <superpixel_count>
- examples:
    - ./SP image test.jpg test_out.jpg 50
    - ./SP video test.mp4 test_out.mp4 50

# build python module
- make module *(in root directory of the repo)*  
  this creates pybuild/boruvka_superpixel.*.so, which is to be imported from
  python

# test python module
1. cd pybuild
2. ./boruvkasupix.py <input_img_path> <output_img_path> <n_supix>  
   example:
	- ./boruvkasupix.py test.jpg test_out.jpg 100
    
# test 2D 3D on DAVIS
1. cd src
2. ./boruvkasupix2D.py <input_folder> <output_folder> <n_supix>  
   example:
	- python3 boruvkasupix2D.py ~/DAVIS/JPEGImages/480p/bmx-bumps/ bmx_bumps_out_2d_40 40
3. ./boruvkasupix3D.py <input_folder> <output_folder> <n_supix>  
   example:
	- python3 boruvkasupix3D.py ~/DAVIS/JPEGImages/480p/bmx-bumps/ bmx_bumps_out_3d_10 10
4. ./boruvkasupix2D_3D.py <input_image_folder> <2D_folder> <3D_folder> <output_video_path>  
   example:
	- python3 boruvkasupix2D_3D.py ~/DAVIS/JPEGImages/480p/bmx-bumps/ bmx_bumps_out_2d_40 bmx_bumps_out_3d_8 bmx_bumps_out_2d_40_3d_10.mp4

# library interface
- c++ and python
- Data types supported: uint8, uint16, int8, int16, int32, float32, float64.  
  In all cases the internal data type is float32 as a compromise between
  precision and memory use.  The data type of the arrays `feature` and
  `border_strength` in `build_2d()` or `build_3d()` should be the same, but 
  it can be independent of the data type of `data` in `average()`.

# algorithm
- The distance between neighboring pixels is the L1 norm of the feature vector
  difference.  In the first 4 iterations this is multiplied by the minimum of
  the `border_strength` on the two pixels, but at least 1.

