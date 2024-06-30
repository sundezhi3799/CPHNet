import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt


# convert the mask from the txt file(annotation_path is path of txt file) to array of points making that mask.
def generate_points(annotation_path='', size=1080):
    labels = []  # this will store labels
    # we are assuming that the image is of dimension (size,size). then you have annotated it.
    with open(annotation_path, "r") as file:
        points = []
        for line in file:
            label, lis = line.split()[0], line.split()[1:]
            labels.append(label)
            lis = list(map(float, lis))
            for i in range(len(lis)):
                lis[i] = int(lis[i] * size)
            newlis = []
            i = 0
            while (i < len(lis)):
                # appendint the coordinates as a tuple (x,y)
                newlis.append((lis[i], lis[i + 1]))
                i += 2
            points.append(newlis)
        return labels, points


def create_edge_area(size, distance):
    matrix = np.zeros((size, size), dtype=int)  # 创建全0矩阵
    for i in range(size):
        for j in range(size):
            distance_to_edge = min(i, j, size - 1 - i, size - 1 - j)
            if distance_to_edge < distance:
                matrix[i, j] = 1
    return matrix

def matching_nu_mem(label1,mask1,label2,mask2,matrix):
    if not (mask1.any() and mask2.any()):
        return False
    elif label2 and not label1:
        if (mask1 * matrix).sum() or (mask2 * matrix).sum():
            return False

        elif (mask1 * mask2 == mask2).all():
            # print(i,j)
            return True
            # plt.imshow(mask1)
            # plt.show()
        else:
            return False
    elif label1 and not label2:
        if (mask1 * matrix).sum() or (mask2 * matrix).sum():
            return False
        elif (mask1 * mask2 == mask1).all():
            # print(i,j)
            return True
        else:
            return False
    else:
        return False
# the below function convert the boundary coordinates to mask array (it shows mask if you pass 1 at show)
# the mask array is required when we want to augument the mask also using albumentation
def convert_boundary_to_mask_array(labels1, points1, labels2, points2, size=1080):
    # Create a new image with the same size as the desired mask
    distance = 5
    matrix = create_edge_area(size, distance)
    mask_arrays = []
    for i, label1 in enumerate(labels1):
        for j, label2 in enumerate(labels2):
            boundary_coord1 = points1[i]
            boundary_coord2 = points2[j]
            mask1 = Image.new("L", (size, size), 0)
            mask2 = Image.new("L", (size, size), 0)
            draw1 = ImageDraw.Draw(mask1)
            draw2 = ImageDraw.Draw(mask2)
            if eval(label2) and not eval(label1):
                draw2.polygon(boundary_coord2, fill=1)
                draw1.polygon(boundary_coord1, fill=1)
                mask_array1 = np.array(mask1)
                mask_array2 = np.array(mask2)
                if (mask_array1 * matrix).sum() or (mask_array2 * matrix).sum():
                    continue
                if (mask_array1 * mask_array2 == mask_array2).all():
                    # print(i,j)
                    draw1.polygon(boundary_coord2, fill=2)
                    mask_arrays.append(np.array(mask1))
                    # plt.imshow(mask1)
                    # plt.show()
            elif eval(label1) and not eval(label2):
                draw2.polygon(boundary_coord2, fill=1)
                draw1.polygon(boundary_coord1, fill=1)
                mask_array1 = np.array(mask1)
                mask_array2 = np.array(mask2)
                if (mask_array1 * matrix).sum() or (mask_array2 * matrix).sum():
                    continue
                if (mask_array1 * mask_array2 == mask_array1).all():
                    # print(i,j)
                    draw2.polygon(boundary_coord1, fill=2)
                    mask_arrays.append(np.array(mask2))
                    # plt.imshow(mask2)
                    # plt.show()
            # elif eval(label2) and not eval(label1):
            #     draw.polygon(boundary_coord1, fill=1)
            #     draw.polygon(boundary_coord2, fill=0)
            # Convert the mask image to a numpy array
            # mask_array = np.array(mask1) * 255
            # plt.imshow(mask_array)
            # plt.show()
            # mask_arrays.append(mask_array)
    return mask_arrays


# function that takes mask path (yolov8 seg txt file) and return mask of an image (shape of mask == shape of image)
def generate_mask(annotation_paths=['', ''], size=1080):
    path1, path2 = annotation_paths
    # pass show=1 for showing the generated mask
    # firstly we generate the points (coordinates) from the annotations
    labels1, points1 = generate_points(path1, size)
    labels2, points2 = generate_points(path2, size)
    # once we get the points we will now generate the mask image from these points (binary mask image (black/white))
    # mask is represented by white and ground is represented as black
    mask_arrays = convert_boundary_to_mask_array(labels1, points1, labels2, points2, size)
    return mask_arrays


# <---------- Helper Functions Ends here ------------------------------------------------------------->
if __name__ == '__main__':
    image_path = "YOLODataset-all\images\\train\\r02c02f02p01-ch5sk1fk1fl1.png"  # path of the image, change it
    annotation_path1 = "runs\segment\predict4\labels\\r02c02f02p01-ch5sk1fk1fl1.txt"  # path of the annotation text file, change it
    annotation_path2 = "runs\segment\predict5\labels\\r02c02f02p01-ch1sk1fk1fl1.txt"  # path of the annotation text file, change it
    # The Helper functions below assume that the image size is (size,size).Hence resizing the image.
    # Open the image
    img = Image.open(image_path)
    size = img.size[0]
    # if you want then you can save the resized image by img.save('resized_image.jpg')
    mask_arrays = generate_mask(annotation_paths=[annotation_path1, annotation_path2], size=size)
    pass
