import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from ultralytics import YOLO
import torch
import cv2
from nuclear_membrane_matching import matching_nu_mem, create_edge_area
# from data_path import *
import numpy as np
def extract_positions(directory):
    filenames = os.listdir(directory)
    # filenames=['r02c05f01p01-.png']
    name_set = set()
    for filename in filenames:
        name_parts = filename.split('-')
        if len(name_parts) > 1:
            name_set.add(name_parts[0])

    return name_set

def normalize_image(image):
    return image/image.max()*255

if __name__ == '__main__':
    distance = 5
    size = 1080
    edge_area_matrix = create_edge_area(size, distance)
    model = YOLO('runs/segment/train52/weights/best.pt')
    input_dir="D:\data\cell_images\\20231220\HPMEC"
    dirs=os.listdir(input_dir)
    for di in dirs:
        # if 'HPMEC' in cell_dir:
        #     continue
    # cell_dir = 'D:\data\cell_images\\20230726\\20230726-CELL PAINTING-HPMEC-5%__2023-07-26T14_59_55-Measurement 1\output'
    #     cell_dir = 'D:\data\cell_images\\20230728\\20230728-CELL PAINTING-A549-5%__2023-07-28T14_36_10-Measurement 2\output'
        cell_dir=os.path.join(input_dir,di)+'\output'
        target_dir='D:\data\cell_images\Substructures1221'
        out_dir = os.path.join(target_dir,os.path.basename(os.path.dirname(cell_dir)))
        if di in os.listdir(target_dir):
            continue
        os.makedirs(out_dir, exist_ok=True)
        holes = extract_positions(cell_dir)
        for hole in holes:
            image_id_dna = hole + '-ch1sk1fk1fl1'
            image_path_dna = os.path.join(cell_dir, image_id_dna + '.tiff')
            image_id_ag = hole+ '-ch3sk1fk1fl1'
            image_path_ag = os.path.join(cell_dir, image_id_ag + '.tiff')
            image_id_er = hole + '-ch2sk1fk1fl1'
            image_path_er = os.path.join(cell_dir, image_id_er + '.tiff')
            image_id_mit = hole + '-ch4sk1fk1fl1'
            image_path_mit = os.path.join(cell_dir, image_id_mit + '.tiff')
            image_id_pm = hole + '-ch5sk1fk1fl1'
            image_path_pm = os.path.join(cell_dir, image_id_pm + '.tiff')
            image_id_rna = hole + '-ch6sk1fk1fl1'
            image_path_rna = os.path.join(cell_dir, image_id_rna + '.tiff')

            image_dna = cv2.imread(image_path_dna,cv2.IMREAD_GRAYSCALE)
            image_ag = cv2.imread(image_path_ag,cv2.IMREAD_GRAYSCALE)
            image_er = cv2.imread(image_path_er,cv2.IMREAD_GRAYSCALE)
            image_mit = cv2.imread(image_path_mit,cv2.IMREAD_GRAYSCALE)
            image_pm = cv2.imread(image_path_pm,cv2.IMREAD_GRAYSCALE)
            image_rna = cv2.imread(image_path_rna,cv2.IMREAD_GRAYSCALE)

            image_dap = cv2.merge((image_dna, image_ag, image_pm))

            predict_dna = model.predict(image_path_dna, save=False, save_txt=False, save_crop=False)
            predict_dap = model.predict(image_dap, save=False, save_txt=False, save_crop=False)
            if predict_dna[0] and predict_dap[0]:
                # masks_dna = predict_dna[0].masks.data.cpu()
                # masks_dap = predict_dap[0].masks.data.cpu()
                points_dna = predict_dna[0].masks.xy
                points_dap = predict_dap[0].masks.xy
                i = 0

                matched_dap=[]
                for dnap in points_dna:
                    if not dnap.size:
                        continue
                    lenth=len(points_dap)
                    for k in range(lenth):
                        if k in matched_dap:
                            continue
                        mask_dap = np.zeros((size, size), dtype='uint8')
                        mask_dna = np.zeros((size, size), dtype='uint8')
                        cv2.fillPoly(mask_dna, np.int32([dnap]), 1)
                        dapp = points_dap[k]
                        if not dapp.size:
                            continue
                        cv2.fillPoly(mask_dap, np.int32([dapp]), 1)
                        matched_mask = matching_nu_mem(0, mask_dap, 1, mask_dna, edge_area_matrix)
                        if matched_mask:
                            matched_dap.append(k)
                            dna=np.array(mask_dna)
                            dap=np.array(mask_dap)
                            i += 1
                            cv2.imwrite(os.path.join(out_dir, hole + '_cell_' + str(i) + '_dna_mask.png'), dna*255)
                            cv2.imwrite(os.path.join(out_dir, hole + '_cell_' + str(i) + '_dap_mask.png'), dap* 255)

                            ag_img = image_ag.copy()
                            ag_img[dap == 0] = 0
                            # ag_img[dna == 1]=0

                            dna_img = image_dna.copy()
                            dna_img[dna == 0] = 0

                            er_img = image_er.copy()
                            er_img[dap == 0] = 0
                            er_img[dna == 1] = 0

                            mit_img = image_mit.copy()
                            mit_img[dap == 0] = 0
                            mit_img[dna == 1] = 0

                            pm_img = image_pm.copy()
                            pm_img[dap == 0] = 0
                            # pm_img[dna == 1] = 0

                            rna_img = image_rna.copy()
                            rna_img[dap == 0] = 0
                            # rna_img[dna == 1] = 0

                            cv2.imwrite(os.path.join(out_dir, hole + '_cell_' + str(i) + '_dna.png'), normalize_image(dna_img))
                            cv2.imwrite(os.path.join(out_dir, hole + '_cell_' + str(i) + '_ag.png'), normalize_image(ag_img))
                            cv2.imwrite(os.path.join(out_dir, hole + '_cell_' + str(i) + '_er.png'), normalize_image(er_img))
                            cv2.imwrite(os.path.join(out_dir, hole + '_cell_' + str(i) + '_mit.png'), normalize_image(mit_img))
                            cv2.imwrite(os.path.join(out_dir, hole + '_cell_' + str(i) + '_pm.png'), normalize_image(pm_img))
                            cv2.imwrite(os.path.join(out_dir, hole + '_cell_' + str(i) + '_rna.png'), normalize_image(rna_img))
                            break
