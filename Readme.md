# Image proccessing course final project
A python code for 1D and 2D signals registration

# run code
1. create python3 venv
2. install requirements `pip install -r req.txt`
3. run 1D registration
   1. open `Task_1d.py` file
   2. change signals `npy` files path with your files path
   3. run code
4. run 2D registration
   1. open `Task_2d.py` file
   2. change signals `jpeg` file path with your file path
   3. run code



# Notes & Limitations
1. Since minimal scale is 0.25, maximal shift should be less than 0.25 * signal length
2. Signal length is important!
3. signals shlud be saved like this
   1. np.savez_compressed('x1.npz',x1=sig1)
   2. np.savez_compressed('x2.npz',x2=sig2)
5. if the signals\imagse are not of the same lengths, they are truncated to the shortest one in their end. This might have an effect
    on the registration result.

### created by : Matan B. and Zeev K.