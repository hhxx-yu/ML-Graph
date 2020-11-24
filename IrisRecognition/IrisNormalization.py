import numpy as np
def IrisNormalization(image, pupil, iris):
    """
    Normalize iris area to a rectangular image of size 64 * 512
    :param image: eye image
    :param pupil: pupil/pupil center and radius
    :param iris: iris/iris boundary center and radius
    :return: normalized iris image
    """
    pupil_height, pupil_width, pupil_r = pupil
    iris_height, iris_width, iris_r = iris

    # Dectected circles for the pupil and the circle of an iris is usually not concentric
    r = int(round(iris_r - np.hypot(pupil_height-iris_height, pupil_width-iris_width)))


    if r < pupil_r + 15:
        r = 80 #set upper bound


    maxh, maxw = image.shape

    r2 = min(pupil_height, pupil_width, maxh-pupil_width, maxw-pupil_height)
    r = min(r, r2)
    max_r = r - pupil_r
  
   # Normalize iris area to a rectangular image of size 64 by 512
    M = 64
    N = 512
    normalized = []
    for y in range(M):
        width_pix = []
        for x in range(N):
            theta = float(2*np.pi*x/N)
            hypotenuse = float(max_r*y/M) + pupil_r
            height = int(round((np.cos(theta) * hypotenuse) + pupil_height))
            width = int(round((np.sin(theta) * hypotenuse) + pupil_width))
            width_pix.append(image[width, height])
        normalized.append(width_pix)

    return np.array(normalized)