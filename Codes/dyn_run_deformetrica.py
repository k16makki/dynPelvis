
#Author: Karim Makki

import os
import argparse
import xml.etree.ElementTree as ET
import glob


import nibabel as nib
import vtk
import numpy as np
from scipy import ndimage

### Example: time python dyn_run_deformetrica.py -dyn /home/karimm/Bureau/input_data/AF_Dyn3D_5SL/AF_Dyn3D_5SL_3dRecbyRegContours_Filled_vtk -m0 /home/karimm/Téléchargements/quadrilateral.vtk -subj AF_Dyn3D_5SL

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='dynPelvis')
    parser.add_argument('-dyn', '--dynamic_set', help='3D dynamic sequence: full path to the reconstructed shape contours, saved as vtk files', type=str, required = True)
    parser.add_argument('-subj', '--subject_name', help='subject name', type=str, required = True)
    parser.add_argument('-m0', '--m0', help='set of points to be tracked, vtk file', type=str, required = True)

    args = parser.parse_args()

    # dyn_vtk_path = './dynamic_set_vtk'
    #
    # if not os.path.exists(dyn_vtk_path):
    #         os.makedirs(dyn_vtk_path)


    dynamicSet = glob.glob(args.dynamic_set+'/'+'*.vtk')
    dynamicSet.sort()


    ### initialization: register the mesh on the first volume

    tree = ET.parse("model.xml")
    root = tree.getroot()

    tree1 = ET.parse("data_set.xml")
    root1 = tree1.getroot()


    #root[2][0][5].text = str(args.source_filename)
    #root1[0][0][0].text = str(args.source_filename)

    root[2][0][5].text = args.m0
    root1[0][0][0].text = str(dynamicSet[1])


    tree.write("model.xml")#, xml_declaration=True, encoding='utf-8')
    tree1.write("data_set.xml")#, xml_declaration=True, encoding='utf-8')

    prefix0 = dynamicSet[1].split('/')[-1].split('.')[0]

    init = 'deformetrica estimate model.xml data_set.xml -p optimization_parameters.xml -v INFO --output '+ './output/'+args.subject_name+'/output_'+prefix0#+'/'

    print(init)

    os.system(init)

    ## Propagation


    for t in range(2,len(dynamicSet)-1):

    #for t in range(2,100):

        neg_prefix = dynamicSet[t-1].split('/')[-1].split('.')[0]

        prefix = dynamicSet[t].split('/')[-1].split('.')[0]

    ## Edit the model and data set xml files

        #input = args.source_filename
        input = './output/'+args.subject_name+'/output_'+neg_prefix+'/DeterministicAtlas__Reconstruction__bladder__subject_subj1.vtk'

        tree = ET.parse("model.xml")
        root = tree.getroot()

        tree1 = ET.parse("data_set.xml")
        root1 = tree1.getroot()

        root[2][0][5].text = input

        root1[0][0][0].text = dynamicSet[t]

        tree.write("model.xml")
        tree1.write("data_set.xml")


        go = 'deformetrica estimate model.xml data_set.xml -p optimization_parameters.xml -v INFO --output '+ './output/'+args.subject_name+'/output_'+prefix

        print(go)

        os.system(go)
