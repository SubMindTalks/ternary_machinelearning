def to_complex_representation(images):
    def pixel_to_complex(pixel):
        return complex(pixel % 16, pixel // 16)  # Arbitrary mapping to complex numbers

    complex_images = np.array([[pixel_to_complex(pixel) for pixel in image.flatten()] for image in images])
    return complex_images.reshape(images.shape[0], images.shape[1], images.shape[2])  # Reshape back
