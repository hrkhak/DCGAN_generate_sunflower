
IMAGE_SIZE = 256

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
####  
  # dictionary of the transformations we defined earlier
available_transformations = {
    'rotate': random_rotation,
    'vertical_flip': vertical_flip,
    'horizontal_flip': horizontal_flip,
    'vertical_and_horizontal_flip': vertical_and_horizontal_flip,
    'TF_crop_pad': TF_crop_pad
}

folder_path = DATASET_FOLDER

# find all files paths from the folder
images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

num_generated_files = 0
while num_generated_files <= NUM_NEW_IMAGES:
    # random image from the folder
    image_path = random.choice(images)
    # read image as an two dimensional array of pixels
    image_to_transform = sk.io.imread(image_path)
    # random num of transformation to apply
    num_transformations_to_apply = random.randint(1, len(available_transformations))

    num_transformations = 0
    transformed_image = None
    while num_transformations <= num_transformations_to_apply:
        # random transformation to apply for a single image
        key = random.choice(list(available_transformations))
        transformed_image = available_transformations[key](image_to_transform)
        num_transformations += 1

        new_file_path = '%s/augmented_image_%s.jpg' % (folder_path, num_generated_files)

        # write image to the disk
        #io.imsave(new_file_path, transformed_image.astype(np.uint8))
        io.imsave(new_file_path, transformed_image)
        num_generated_files += 1
