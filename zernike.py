
import numpy as np
from math import atan2
from numpy import cos, sin, conjugate, sqrt
import cv2
import pylab as pl
from matplotlib import cm

def _slow_zernike_poly(Y,X,n,l):
    def _polar(r,theta):
        x = r * cos(theta)
        y = r * sin(theta)
        return 1.*x+1.j*y

    def _factorial(n):
        if n == 0: return 1.
        return n * _factorial(n - 1)
    y,x = Y[0],X[0]
    vxy = np.zeros(Y.size, dtype=complex)
    index = 0
    for x,y in zip(X,Y):
        Vnl = 0.
        for m in range( int( (n-l)//2 ) + 1 ):
            Vnl += (-1.)**m * _factorial(n-m) /  \
                ( _factorial(m) * _factorial((n - 2*m + l) // 2) * _factorial((n - 2*m - l) // 2) ) * \
                ( sqrt(x*x + y*y)**(n - 2*m) * _polar(1.0, l*atan2(y,x)) )
        vxy[index] = Vnl
        index = index + 1

    return vxy

def zernike_reconstruct(img, radius, D, cof):

    idx = np.ones(img.shape)

    cofy,cofx = cof
    cofy = float(cofy)
    cofx = float(cofx)
    radius = float(radius)    

    Y,X = np.where(idx > 0)
    P = img[Y,X].ravel()
    Yn = ( (Y -cofy)/radius).ravel()
    Xn = ( (X -cofx)/radius).ravel()

    k = (np.sqrt(Xn**2 + Yn**2) <= 1.)
    frac_center = np.array(P[k], np.double)
    Yn = Yn[k]
    Xn = Xn[k]
    frac_center = frac_center.ravel()

    # in the discrete case, the normalization factor is not pi but the number of pixels within the unit disk
    npix = float(frac_center.size)

    reconstr = np.zeros(img.size, dtype=complex)
    accum = np.zeros(Yn.size, dtype=complex)

    for n in range(D+1):
        for l in range(n+1):
            if (n-l)%2 == 0:
                # get the zernike polynomial
                vxy = _slow_zernike_poly(Yn, Xn, float(n), float(l))
                # project the image onto the polynomial and calculate the moment
                a = sum(frac_center * conjugate(vxy)) * (n + 1)/npix
                # reconstruct
                accum += a * vxy
    reconstr[k] = accum
    return reconstr

# if __name__ == '__main__':

#     import cv2
#     import pylab as pl
#     from matplotlib import cm

#     D = 12

#     img = cv2.imread('samp/0.png', 0)

#     rows, cols = img.shape
#     radius = cols//2 if rows > cols else rows//2

#     reconst = zernike_reconstruct(img, radius, D, (rows/2., cols/2.))

#     reconst = reconst.reshape(img.shape)

#     pl.figure(1)
#     pl.imshow(img, cmap=cm.jet, origin = 'upper')
#     pl.figure(2)    
#     pl.imshow(reconst.real, cmap=cm.jet, origin = 'upper')





D = 12
img = cv2.imread('samp/0.png', 0)
rows, cols = img.shape
radius = cols//2 if rows > cols else rows//2

reconst = zernike_reconstruct(img, radius, D, (rows/2., cols/2.))

reconst = reconst.reshape(img.shape)

pl.figure(1)
pl.imshow(img, cmap=cm.jet, origin = 'upper')
pl.figure(2)    
pl.imshow(reconst.real, cmap=cm.jet, origin = 'upper')

cv2.imshow("img",reconst.real)
cv2.waitKey(0)
cv2.destroyAllWindows()