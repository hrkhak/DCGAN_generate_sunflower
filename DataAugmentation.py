
IMAGE_SIZE = 256
NUM_NEW_IMAGES = 1000

def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 20% on the left and 20% on the right
    random_degree = random.uniform(-20, 20)
    return sk.transform.rotate(image_array, random_degree)

def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

def vertical_flip(image_array: ndarray):
    return image_array[::-1, :]

def vertical_and_horizontal_flip(image_array: ndarray):
    h_flip_image = vertical_flip(image_array)
    return horizontal_flip(h_flip_image)

def TF_crop_pad(x, n_pixels=20, pad_mode='edge'):
    """Pad image by n_pixels on each size, then take random crop of same
    original size.
    """
    assert len(x.shape) == 3
    h, w, nc = x.shape

    # First pad image by n_pixels on each side
    padded = sk.util.pad(x, [(n_pixels, n_pixels) for _ in range(2)] + [(0,0)],
        mode=pad_mode)

    # Then take a random crop of the original size
    crops = [(c, 2*n_pixels-c) for c in np.random.randint(0, 2*n_pixels+1, [2])]
    # For channel dimension don't do any cropping
    crops += [(0,0)]
    return sk.transform.resize(sk.util.crop(padded, crops, copy=True), (IMAGE_SIZE, IMAGE_SIZE)) 
