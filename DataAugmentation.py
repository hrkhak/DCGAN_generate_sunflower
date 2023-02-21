
IMAGE_SIZE = 256
NUM_NEW_IMAGES = 1000

def random_rotation(image_array: ndarray):
    # چرخش ۱۵ درصدی به سمت چپ و راست بصورت تصادفی
    random_degree = random.uniform(-15, 15)
    return sk.transform.rotate(image_array, random_degree)

def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

def vertical_flip(image_array: ndarray):
    return image_array[::-1, :]

def vertical_and_horizontal_flip(image_array: ndarray):
    h_flip_image = vertical_flip(image_array)
    return horizontal_flip(h_flip_image)

def TF_crop_pad(x, n_pixels=15, pad_mode='edge'):
    """اضافه کردن پد به اندازه ۱۵ پیکسل و به صورت تصادفی برش تصویر به همین اندازه
    """
    assert len(x.shape) == 3
    h, w, nc = x.shape

    # اولین پیدینگ تصویر
    padded = sk.util.pad(x, [(n_pixels, n_pixels) for _ in range(2)] + [(0,0)],
        mode=pad_mode)

    # کراپ تصویر اصلی
    crops = [(c, 2*n_pixels-c) for c in np.random.randint(0, 2*n_pixels+1, [2])]
    # For channel dimension don't do any cropping
    crops += [(0,0)]
    return sk.transform.resize(sk.util.crop(padded, crops, copy=True), (IMAGE_SIZE, IMAGE_SIZE)) 
