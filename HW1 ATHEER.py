#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
from math import sqrt,exp
import glob


# In[2]:


def plot_before_after(img1, img2, is_magnitude=False):
    """
    Plot images before and after editing alongside.

    Parameters
    ----------
    img1 : numpy.array
        The original image.
    img2 : numpy.array
        The editted image.
    is_magnitude : bool, optional
        Whether to plot the magnitude spectrum or not. The default is False.

    Returns
    -------
    None.

    """
    plt.subplot(121)
    plt.imshow(img1, cmap='gray')
    plt.title('Input Image')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(img2, cmap='gray')
    if is_magnitude:
        plt.title('Magnitude Spectrum')
    else:
        plt.title('Filtered Image')
    plt.xticks([])
    plt.yticks([])
    plt.show()


# In[3]:


def gausskernel(size,k,sigma):
    gausskernel = np.zeros((size,size),np.float32)
    for i in range (size):
        for j in range (size):
            norm = math.pow(i-k,2) + pow(j-k,2)
            gausskernel[i,j] = math.exp(-norm/(2*math.pow(sigma,2)))/2*math.pi*pow(sigma,2)
    sum = np.sum(gausskernel)
    kernel = gausskernel/sum 
    return kernel
def mygaussFilter(img_gray,kernel):
    h,w = img_gray.shape
    k_h,k_w = kernel.shape
    for i in range(int(k_h/2),h-int(k_h/2)):
        for j in range(int(k_h/2),w-int(k_h/2)):
            sum = 0
            for k in range(0,k_h):
                for l in range(0,k_h):
                    sum += img_gray[i-int(k_h/2)+k,j-int(k_h/2)+l]*kernel[k,l]
            img_gray[i,j] = sum
    return img_gray


# In[4]:


def gaussHighPassFilter(shape, radius=10):  # Gaussian Highpass Filter 
        # Gaussian filter:# Gauss = 1/(2*pi*s2) * exp(-(x**2+y**2)/(2*s2))
        u, v = np.mgrid[-1:1:2.0/shape[0], -1:1:2.0/shape[1]]
        D = np.sqrt(u**2 + v**2)
        D0 = radius / shape[0]
        kernel = 1 - np.exp(- (D ** 2) / (2 *D0**2))
        return kernel


# In[5]:


def gaussLowPassFilter(shape, radius=10):  # Gaussian low pass filter
    # Gaussian filter:# Gauss = 1/(2*pi*s2) * exp(-(x**2+y**2)/(2*s2))
    u, v = np.mgrid[-1:1:2.0/shape[0], -1:1:2.0/shape[1]]
    D = np.sqrt(u**2 + v**2)
    D0 = radius / shape[0]
    kernel = np.exp(- (D ** 2) / (2 *D0**2))
    return kernel


# In[6]:


def dft2Image(image):  #Optimal extended fast Fourier transform
        # Centralized 2D array f (x, y) * - 1 ^ (x + y)
        mask = np.ones(image.shape)
        mask[1::2, ::2] = -1
        mask[::2, 1::2] = -1
        fImage = image * mask  # f(x,y) * (-1)^(x+y)
        # Optimal DFT expansion size
        rows, cols = image.shape[:2]  # The height and width of the original picture
        rPadded = cv2.getOptimalDFTSize(rows)  # Optimal DFT expansion size
        cPadded = cv2.getOptimalDFTSize(cols)  # For fast Fourier transform
        # Edge extension (complement 0), fast Fourier transform
        dftImage = np.zeros((rPadded, cPadded, 2), np.float32)  # Edge expansion of the original image
        dftImage[:rows, :cols, 0] = fImage  # Edge expansion, 0 on the lower and right sides
        cv2.dft(dftImage, dftImage, cv2.DFT_COMPLEX_OUTPUT)  # fast Fourier transform 
        return dftImage


# In[7]:


def imgHPFilter(image, D0=50):  #Image high pass filtering
        rows, cols = image.shape[:2]  # The height and width of the picture
        # fast Fourier transform 
        dftImage = dft2Image(image)  # Fast Fourier transform (rPad, cPad, 2)
        rPadded, cPadded = dftImage.shape[:2]  # Fast Fourier transform size, original image size optimization
 # Construct Gaussian low pass filter
        hpFilter = gaussHighPassFilter((rPadded, cPadded), radius=D0)  # Gaussian Highpass Filter 

        # Modify Fourier transform in frequency domain: Fourier transform point multiplication high pass filter
        dftHPfilter = np.zeros(dftImage.shape, dftImage.dtype)  # Size of fast Fourier transform (optimized size)
        for j in range(2):
            dftHPfilter[:rPadded, :cPadded, j] = dftImage[:rPadded, :cPadded, j] * hpFilter
# The inverse Fourier transform is performed on the high pass Fourier transform and only the real part is taken
        idft = np.zeros(dftImage.shape[:2], np.float32)  # Size of fast Fourier transform (optimized size)
        cv2.dft(dftHPfilter, idft, cv2.DFT_REAL_OUTPUT + cv2.DFT_INVERSE + cv2.DFT_SCALE)

        # Centralized 2D array g (x, y) * - 1 ^ (x + y)
        mask2 = np.ones(dftImage.shape[:2])
        mask2[1::2, ::2] = -1
        mask2[::2, 1::2] = -1
        idftCen = idft * mask2  # g(x,y) * (-1)^(x+y)

        # Intercept the upper left corner, the size is equal to the input image
        result = np.clip(idftCen, 0, 255)  # Truncation function, limiting the value to [0255]
        imgHPF = result.astype(np.uint8)
        imgHPF = imgHPF[:rows, :cols]
        return imgHPF


# In[8]:


def distance(point1,point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def idealFilterLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 1
    return base

def idealFilterHP(D0,imgShape):
    base = np.ones(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 0
    return base


# In[9]:


def dft(img):
    """
    Transform the image into frequency domain.

    Parameters
    ----------
    img : numpy.array
        The original image.

    Returns
    -------
    shifted_transformed_img : numpy.array
        The transformed image.
    magnitude_spectrum : numpy.array
        Magnitude spectrum of the image.

    """
    transformed_img = np.fft.fft2(img)
    shifted_transformed_img = np.fft.fftshift(transformed_img)
    magnitude_spectrum = np.log(1 + np.abs(shifted_transformed_img))
    return shifted_transformed_img, magnitude_spectrum


# In[10]:


def idft(img):
    """
    Transform image from frequency domain into spatial domain.

    Parameters
    ----------
    img : numpy.array
        An image in frequency domain.

    Returns
    -------
    filtered_img : numpy.array
        The image in spatial domain.

    """
    filtered_img = np.fft.ifftshift(img)
    filtered_img = np.fft.ifft2(
        filtered_img).real.clip(0, 255).astype(np.uint8)
    return filtered_img


# In[11]:


def gaussian_lowpass_filter(shape=(3, 3), sigma=0.5):
    """
    Create a lowpass Gaussian filter.

    Parameters
    ----------
    shape : tuple, optional
        Size of the filter. The default is (3, 3).
    sigma : float, optional
        The Gaussian parameter sigma. The default is 0.5.

    Returns
    -------
    h : numpy.array
        The filter.

    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma ** 2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


# In[12]:


def gaussian_band_reject(shape=(3, 3), sigma=0.5, w=1):
    """
    Create a band reject Gaussian filter.

    Parameters
    ----------
    shape : tuple, optional
        Size of the filter. The default is (3, 3).
    sigma : float, optional
        The Gaussian parameter sigma. The default is 0.5.
    w : float, optional
        The bandwidth. The default is 1.

    Returns
    -------
    h : numpy.array
        The filter.

    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = 1 - np.exp(-(((x * x + y * y) - sigma**2) /
                     (np.sqrt(x * x + y * y) * w))**2)
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


# In[13]:


def homomorphic_filtering(img, rh, rl, cutoff, c=1):
    """
    Apply homomorphic filter.

    Parameters
    ----------
    img : numpy.array
        The original image.
    rh : float
        Gamma high parameter.
    rl : float
        Gamma low parameter.
    cutoff : float
        Cutoff value.
    c : float, optional
        The multiplier c in the filter. The default is 1.

    Returns
    -------
    numpy.array
        The filtered image.

    """
    img = np.float32(img)
    img = img / 255
    rows, cols = img.shape

    img_log = np.log(img + 1)

    img_fft_shift, _ = dft(img_log)

    DX = cols / cutoff
    G = np.ones((rows, cols))
    for i in range(rows):
        for j in range(cols):
            G[i][j] = ((rh - rl) * (1 - np.exp(-c * ((i - rows / 2) ** 2 +
                                                     (j - cols / 2)**2) /
                                               (2 * DX**2)))) + rl

    result_filter = G * img_fft_shift

    result_interm = np.real(np.fft.ifft2(np.fft.ifftshift(result_filter)))

    return np.exp(result_interm)


# In[14]:


def image1():
    """
    Edit image1.jpg.

    Returns
    -------
    None.

    """
    img = cv2.imread("D:/image processing/images/image1.jpg", 0)
    img_blur = cv2.blur(img,(5,5))
    plot_before_after(img, img_blur)    


# In[15]:


def image2():
    """
    Edit image2.jpg.

    Returns
    -------
    None.

    """
    img = cv2.imread("D:/image processing/images/image2.jpg", 0)
    filtered_img = cv2.resize(img, (0, 0), fx=1.5, fy=1.5)
    height, width = img.shape[:2]
    center = (width/2, height/2)
    #the above center is the center of rotation axis
    # use cv2.getRotationMatrix2D() to get the rotation matrix
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=90, scale=1)
    # Rotate the image using cv2.warpAffine
    rotated_image = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(width, height))
    # get tx and ty values for translation
    # you can specify any value of your choice
    tx, ty = width / 4, height / 4
    # create the translation matrix using tx and ty, it is a NumPy array
    translation_matrix = np.array([
        [1, 0, tx],
        [0, 1, ty]
        ], dtype=np.float32)
    # apply the translation to the image
    translated_image = cv2.warpAffine(src=img, M=translation_matrix, dsize=(width, height))
    image_median = cv2.medianBlur(img,5)
    plot_before_after(img, filtered_img)
    plot_before_after(img, rotated_image)
    plot_before_after(img, translated_image)
    plot_before_after(img, image_median)


# In[16]:


def image3():
    """
    Edit image3.jpg.

    Returns
    -------
    None.

    """
    img = cv2.imread("D:/image processing/images/image3.jpg", 0)
    # remove noise
    filtered_img = cv2.GaussianBlur(img,(5,5),0)
    # convolute with proper kernels
    laplacian = cv2.Laplacian(filtered_img,cv2.CV_64F)
    plot_before_after(img, laplacian)    


# In[17]:


def image4():
    """
    Edit image4.jpg.

    Returns
    -------
    None.

    """
    img = cv2.imread("D:/image processing/images/image4.jpg", 0)
    # remove noise
    filtered_img = cv2.GaussianBlur(img,(7,7),0)
    # convolute with proper kernels
    sobelx = cv2.Sobel(filtered_img,cv2.CV_64F,1,0,ksize=7)  # x
    sobely = cv2.Sobel(filtered_img,cv2.CV_64F,0,1,ksize=7)  # y
    plot_before_after(img, sobelx)  
    plot_before_after(img, sobely)   


# In[18]:


def image5():
    """
    Edit image5.jpg.

    Returns
    -------
    None.

    """
    img = cv2.imread("D:/image processing/images/image5.jpg", 0)
    shifted_transformed_img, magnitude_spectrum = dft(img)
    plot_before_after(img, magnitude_spectrum, True)
    filtered_img = idft(shifted_transformed_img)
    plot_before_after(img, filtered_img)
    inv_center = np.fft.ifftshift(magnitude_spectrum)
    plot_before_after(img, inv_center)
    LowPass = idealFilterLP(50,img.shape)
    plot_before_after(img, LowPass)
    HighPass = idealFilterHP(50,img.shape)
    plot_before_after(img, HighPass)


# In[19]:


def image6():
    """
    Edit image6.jpg.

    Returns
    -------
    None.

    """

    img = cv2.imread("D:/image processing/images/image6.jpg")
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img_g = img_gray.copy()
    for k in (3, 5, 7, 9):
        size = 2*k+1
        kernel = gausskernel(size,k,1.5)
        #print(kernel)
        img_B,img_G,img_R = cv2.split(img)
        img_gauss_B = mygaussFilter(img_B,kernel)
        img_gauss_G = mygaussFilter(img_G,kernel)
        img_gauss_R = mygaussFilter(img_R,kernel)
        img_gauss = cv2.merge([img_gauss_B,img_gauss_G,img_gauss_R])
        #img_comp = np.hstack((img,img_gauss))
        plot_before_after(img, img_gauss)


# In[20]:


def im1():
    """
    Edit image 1.jpg.

    Returns
    -------
    None.

    """
    img = cv2.imread('D:/image processing/images/1.jpg', 0)
    shifted_transformed_img, magnitude_spectrum = dft(img)

    plot_before_after(img, magnitude_spectrum, True)

    w, h = img.shape
    mask = np.ones(img.shape, dtype=np.uint8)

    for i in range(w):
        if np.mean(magnitude_spectrum[i, :]) >= 9.:
            # magnitude_spectrum[i, :] = 0
            mask[i, :] = 0

    for j in range(h):
        if np.mean(magnitude_spectrum[:, j]) >= 9.:
            # magnitude_spectrum[:, j] = 0
            mask[:, j] = 0
    mask[w // 2 - 5:w // 2 + 5, h // 2 - 5:h // 2 + 5] = 1

    plt.imshow(mask * magnitude_spectrum, cmap='gray')
    plt.show()

    filtered_img = idft(mask * shifted_transformed_img)
    filtered_img = cv2.GaussianBlur(filtered_img, (5, 5), 0)

    plot_before_after(img, filtered_img)


# In[21]:


def im2():
    """
    Edit image 2.jpg.

    Returns
    -------
    None.

    """
    img = cv2.imread("D:/image processing/images/2.jpg", 0)
    shifted_transformed_img, magnitude_spectrum = dft(img)

    w, h = img.shape
    mask = np.ones(img.shape)
    mask[0:w // 2 - 5, h // 2 - 2:h // 2 + 2] = 0
    mask[w // 2 + 5:w, h // 2 - 2:h // 2 + 2] = 0

    plt.imshow(magnitude_spectrum * mask, cmap='gray')
    plt.show()

    masked_fourier = shifted_transformed_img * mask
    filtered_img = idft(masked_fourier)
    filtered_img = cv2.blur(filtered_img, (2, 2))

    plot_before_after(img, filtered_img)


# In[22]:


def im3():
    """
    Edit image 3.jpg.

    Returns
    -------
    None.

    """
    img = cv2.imread("D:/image processing/images/3.jpg", 0)
    shifted_transformed_img, magnitude_spectrum = dft(img)
    w, h = img.shape
    mask = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    cv2.circle(mask, (96, 117), 10, (0, 0, 0), -1)
    cv2.circle(mask, (42, 85), 10, (0, 0, 0), -1)
    cv2.circle(mask, (204, 178), 10, (0, 0, 0), -1)
    cv2.circle(mask, (258, 210), 10, (0, 0, 0), -1)

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    plt.imshow(magnitude_spectrum * mask, cmap='gray')
    plt.show()
    masked_fourier = shifted_transformed_img * mask
    filtered_img = idft(masked_fourier)
    plot_before_after(img, filtered_img)


# In[23]:


def im4():
    """  Edit image 4.jpg. Returns    -------    None.    """
    img = cv2.imread("D:/image processing/images/4.jpg", 0)
    rows, cols = img.shape[:2]  # The height and width of the picture
    imgHPF = imgHPFilter(img, D0=50)
    imgThres = np.clip(imgHPF, 0, 1)
    plot_before_after(img, imgHPF)
    plot_before_after(img, imgThres)
    shifted_transformed_img, magnitude_spectrum = dft(img)
    plot_before_after(img, magnitude_spectrum, True)
    w, h = img.shape
    mask = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    cv2.line(mask, (155, 150), (155, 350), (0, 0, 0), 8)
    cv2.line(mask, (205, 150), (205, 370), (0, 0, 0), 8)
    cv2.line(mask, (240, 110), (240, 400), (0, 0, 0), 8)
    cv2.line(mask, (0, 265), (350, 265), (0, 0, 0), 8)
    cv2.line(mask, (375, 265), (h, 265), (0, 0, 0), 8)
    cv2.line(mask, (290, 110), (290, 400), (0, 0, 0), 8)
    cv2.line(mask, (410, 110), (410, 400), (0, 0, 0), 8)
    cv2.line(mask, (510, 110), (510, 400), (0, 0, 0), 8)
    cv2.line(mask, (345, 70), (150, 255), (0, 0, 0), 8)
    cv2.line(mask, (370, 505), (590, 270), (0, 0, 0), 8)
    cv2.line(mask, (445, 90), (445, 460), (0, 0, 0), 8)
    cv2.line(mask, (540, 150), (540, 350), (0, 0, 0), 8)
    cv2.line(mask, (0, 190), (h, 190), (0, 0, 0), 8)
    cv2.line(mask, (0, 340), (h, 340), (0, 0, 0), 8)
    cv2.line(mask, (0, 305), (h, 305), (0, 0, 0), 8)
    cv2.line(mask, (0, 220), (h, 220), (0, 0, 0), 8)
    cv2.circle(mask, (270, 440), 10, (0, 0, 0), -1)
    cv2.circle(mask, (490, 160), 10, (0, 0, 0), -1)
    cv2.rectangle(mask, (0, 0), (h, 70), (0, 0, 0), -1)
    cv2.rectangle(mask, (0, 480), (h, w), (0, 0, 0), -1)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask[0:w // 2 - 5, h // 2 - 2:h // 2 + 2] = 0
    mask[w // 2 + 5:w, h // 2 - 2:h // 2 + 2] = 0
    plt.imshow(20 * magnitude_spectrum * mask, cmap='gray')
    plt.show()
    masked_fourier = shifted_transformed_img * mask
    filtered_img = idft(masked_fourier)
    plot_before_after(img, filtered_img)


# In[24]:


def im5678910():
    """
    Edit image 5, 6, 7, 8, 9, 10.jpg. Returns -------None.    """
    # High frequency filtering + histogram equalization
    path= "D:/image processing/images/23/*.jpg"
    image_number=1
    for file in glob.glob(path):
        print (file)
        image =cv2.imread(file, 0)
        rows, cols = image.shape[:2]  # The height and width of the picture
        print(rows, cols)
        # fast Fourier transform 
        dftImage = dft2Image(image)  # Fast Fourier transform (rPad, cPad, 2)
        rPadded, cPadded = dftImage.shape[:2]  # Fast Fourier transform size, original image size optimization
        # Construct Gaussian low pass filter
        hpFilter = gaussHighPassFilter((rPadded, cPadded), radius=40)  # Gaussian Highpass Filte
        # Modify Fourier transform in frequency domain: Fourier transform point multiplication low-pass filter
        dftHPfilter = np.zeros(dftImage.shape, dftImage.dtype)  # Size of fast Fourier transform (optimized size)
        for j in range(2):
            dftHPfilter[:rPadded, :cPadded, j] = dftImage[:rPadded, :cPadded, j] * hpFilter
            # The inverse Fourier transform is performed on the high pass Fourier transform and only the real part is taken
            idft = np.zeros(dftImage.shape[:2], np.float32)  # Size of fast Fourier transform (optimized size)
            cv2.dft(dftHPfilter, idft, cv2.DFT_REAL_OUTPUT + cv2.DFT_INVERSE + cv2.DFT_SCALE)
        # Centralized 2D array g (x, y) * - 1 ^ (x + y)
        mask2 = np.ones(dftImage.shape[:2])
        mask2[1::2, ::2] = -1
        mask2[::2, 1::2] = -1
        idftCen = idft * mask2  # g(x,y) * (-1)^(x+y)
        # Intercept the upper left corner, the size is equal to the input image
        result = np.clip(idftCen, 0, 255)  # Truncation function, limiting the value to [0255]
        imgHPF = result.astype(np.uint8)
        imgHPF = imgHPF[:rows, :cols]
        # # =======High frequency enhanced filtering===================
        k1 = 0.5
        k2 = 0.75
        # Modify Fourier transform in frequency domain: Fourier transform point multiplication low-pass filter
        hpEnhance = np.zeros(dftImage.shape, dftImage.dtype)  # Size of fast Fourier transform (optimized size)
        for j in range(2):
            hpEnhance[:rPadded, :cPadded, j] = dftImage[:rPadded, :cPadded, j] * (k1 + k2*hpFilter)
            # The inverse Fourier transform is performed on the high pass Fourier transform and only the real part is taken
            idft = np.zeros(dftImage.shape[:2], np.float32)  # Size of fast Fourier transform (optimized size)
            cv2.dft(hpEnhance, idft, cv2.DFT_REAL_OUTPUT + cv2.DFT_INVERSE + cv2.DFT_SCALE)
        # Centralized 2D array g (x, y) * - 1 ^ (x + y)
        mask2 = np.ones(dftImage.shape[:2])
        mask2[1::2, ::2] = -1
        mask2[::2, 1::2] = -1
        idftCen = idft * mask2  # g(x,y) * (-1)^(x+y)
        # Intercept the upper left corner, the size is equal to the input image
        result = np.clip(idftCen, 0, 255)  # Truncation function, limiting the value to [0255]
        imgHPE= result.astype(np.uint8)
        imgHPE = imgHPE[:rows, :cols]
        # =======Histogram equalization===================
        imgEqu = cv2.equalizeHist(imgHPE)  # Use CV2 Equalizehist completes histogram equalization transformation
        plot_before_after(image, imgHPE)
        plot_before_after(image, imgEqu)
    image_number +=1


# In[25]:


def im1112():
    """
    Edit image 12.jpg. Returns   -------  None.    """
        # (1) Read original image
    path= "D:/image processing/images/24/*.jpg"
    image_number=1
    for file in glob.glob(path):
        print (file)
        imgGray =cv2.imread(file, 0)
        rows, cols = imgGray.shape[:2]  # The height and width of the picture
        # (2) Fast Fourier transform
        dftImage = dft2Image(imgGray)  # Fast Fourier transform (rPad, cPad, 2)
        rPadded, cPadded = dftImage.shape[:2]  # Fast Fourier transform size, original image size optimization
        print("dftImage.shape:{}".format(dftImage.shape))
        D0 = [10, 30, 60, 80, 100]  # radius
        for k in range(5):
            # (3) Construct Gaussian low pass filter
            lpFilter = gaussLowPassFilter((rPadded, cPadded), radius=D0[k])

            # (5) Modify Fourier transform in frequency domain: Fourier transform point multiplication low-pass filter
            dftLPfilter = np.zeros(dftImage.shape, dftImage.dtype)  # Size of fast Fourier transform (optimized size)
            for j in range(2):
                dftLPfilter[:rPadded, :cPadded, j] = dftImage[:rPadded, :cPadded, j] * lpFilter

            # (6) The inverse Fourier transform is performed on the low-pass Fourier transform, and only the real part is taken
            idft = np.zeros(dftImage.shape[:2], np.float32)  # Size of fast Fourier transform (optimized size)
            cv2.dft(dftLPfilter, idft, cv2.DFT_REAL_OUTPUT + cv2.DFT_INVERSE + cv2.DFT_SCALE)

            # (7) Centralized 2D array g (x, y) * - 1 ^ (x + y)
            mask2 = np.ones(dftImage.shape[:2])
            mask2[1::2, ::2] = -1
            mask2[::2, 1::2] = -1
            idftCen = idft * mask2  # g(x,y) * (-1)^(x+y)

            # (8) Intercept the upper left corner, the size is equal to the input image
            result = np.clip(idftCen, 0, 255)  # Truncation function, limiting the value to [0255]
            imgLPF = result.astype(np.uint8)
            imgLPF = imgLPF[:rows, :cols]
            plot_before_after(imgGray, imgLPF)
           
    image_number +=1


# In[26]:


def __main__(num):
    name = "image{}".format(num)
    globals()[name]()


if __name__ == "__main__":
    try:
        __main__(int(sys.argv[1]))
    except ValueError:
        print(
            "PLEASE ENTER IMAGE NUMBER AFTER PROGRAM NAME(A NUMBER IN",
            "RANGE [1, 6]).")
        print(
            "or PLEASE ENTER im NUMBER AFTER PROGRAM NAME(A NUMBER IN",
            "RANGE [1, 4]).")
        print(
            "or PLEASE ENTER {im5678910}" 
            " or PLEASE ENTER {im5678910}.")


# In[28]:


im4()


# In[ ]:





# In[ ]:





# In[ ]:




