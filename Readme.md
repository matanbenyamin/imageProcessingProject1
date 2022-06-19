# Image proccessing course final project
A python code for 1D and 2D signals registration


# run from command line
1. create python3 venv
2. install requirements `pip install -r requirements.txt`
3. run 1D registration
   1.Iterative: run  `python iterative-1d-gm.py x1.npz x2.npz` where x1.npz and x2.npz are your signals saved as mentioned in the readme
   2. Multiscale: run `python Multiscale-1d-gm.py x1.npz x2.npz` where x1.npz and x2.npz are your signals saved as mentioned in the readme
4. run 2D registration
   1. run `python Iterative-2d-gm.py im1.bmp im2.bmp` where im1.bmp and im2.bmp are your images saved as mentioned in the readme
   2. run `python Multiscale-2d-gm.py im1.bmp im2.bmp` where im1.bmp and im2.bmp are your images saved as mentioned in the readme

## Importannt - automatic sigma selection! 
1. A specific sigmal was selected for smoothing the signal and images
2. However, there is possibilty to optimize sigma by addind "auto" to the end of the command
3. This will automatically select the sigma for smoothing the signal and images, by minimizing the error between sig1 and shifted sig2
4. example
   1. `python Iterative-1d-gm.py x1.npz x2.npz auto`


# Notes & Limitations
1. Since minimal scale is 0.25, maximal shift should be less than 0.25 * signal length
2. Signal length is important!
3. signals shlud be saved like this
   1. np.savez_compressed('x1.npz',x1=sig1)
   2. np.savez_compressed('x2.npz',x2=sig2)
4. if the signals\imagse are not of the same lengths, they are truncated to the shortest one in their end. This might have an effect
    on the registration result.
5. generateSignals is a function that generates random signals. You can use it to generate your own signals.

# Performance Analysis
The attached Tests.py  file contains the performance analysis of the registration.
It scans the performance of the registration on a range of parameters.
The performance is measured as the mean of the mean of the registration error.
It is performed on random signals and images, but can be performed on custom signals and images, following the instructions in the file.
Analysis can also be found in the folder as htmls ('1D.html' and '2D.html')

### created by : Matan B. and Zeev K.