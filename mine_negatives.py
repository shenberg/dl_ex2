import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets.folder import default_loader

import argparse
import os

import q1
import q2
import models
import utils

def get_image_patches(image_tensor, boxes):
    if len(boxes) == 0:
        return []
    patches = []
    # TODO: *2 is yuck
    for box in boxes.round().long()*2:
        # index image tensor by y, then x (row, col)
        patches.append(image_tensor[:, box[1]:box[3], box[0]:box[2]])
    return patches


def scan_multiple_scales(net, image, scales, threshold=0.1):
    total_matches = []
    image_size_t = torch.Tensor([image.size[0], image.size[1]])
    # impose additional condition: destination size is even. helps deal with annoying rounding issues
    # that pop up because pytorch has no "SAME" rounding a la tensorflow
    dest_sizes = [(image_size_t*scale/2).round()*2 for scale in scales]
    
    for scale, dest_size in zip(scales, dest_sizes):
        if any(dest_size < 12):
            print("image too small for given scale: {}, {}, {}".format(scale, image_size_t, dest_size))
            continue

        transform = transforms.Compose([transforms.Scale(dest_size.int()),
                                      transforms.ToTensor()])
        double_transform = transforms.Compose([transforms.Scale(dest_size.int()*2),
                                      transforms.ToTensor()])
        image_tensor = transform(image)
        double_image_tensor = double_transform(image)

        # scan, nms on results
        results_for_scale = q2.nms(q2.scan(net, image_tensor, threshold).float())

        # skip no matches for this scale 
        if len(results_for_scale) == 0:
            continue
        patches = get_image_patches(double_image_tensor, results_for_scale)

        # now calculate exact scale for W,H since we rounded
        actual_size = image_tensor.size()

        # actual size is 3xHxW, reverse order here
        source_size = torch.Tensor([actual_size[-1], actual_size[-2]])
        
        real_scale = image_size_t / source_size

        # create [w,h,w,h,1] tensor for scaling the results (x1,y1,x2,y2,score)
        scale_tensor = torch.Tensor([real_scale[1], real_scale[0]]*2 + [1])

        total_matches.append((results_for_scale*scale_tensor, patches, scale))

    return total_matches

def scan_for_negatives(net, negative_path, threshold):
    image = default_loader(negative_path)
    results = scan_multiple_scales(net, image, utils.scan_scales, threshold)
    # results is a list of (boxes, matching patches, scale of original image) tuples
    total_patches = []
    total_matches = []
    for matches, patches, scale in results:
        total_patches.extend(patches)
        total_matches.extend(matches)

    return total_patches, total_matches

def main():

    main_arg_parser = argparse.ArgumentParser(description="options")
    main_arg_parser.add_argument("--voc-path", help="path to VOC2007 directory (should contain JPEGImages, Imagesets dirs)", default="EX2_data/VOC2007")
    main_arg_parser.add_argument("--checkpoint", help="checkpoint file (from q1.py) path", default="q1_batchnorm2d_400epochs_threshold_0.2.pth.tar")
    main_arg_parser.add_argument("--output-path", help="path to write serialized negative patches sets", default="EX2_data/negative_mines")
    # determined default by looking at precision-recall curve, choosing the most precision for >99% recall.
    main_arg_parser.add_argument("-t", "--threshold", help="positive cutoff", default=0.2, type=float)

    args = main_arg_parser.parse_args()

    net = models.Net12FCN()
    net.load_from_net12(args.checkpoint)
    net.eval()

    voc_path = "EX2_data/VOC2007"
    negative_files = q1.get_voc2007_negatives(voc_path)

    for negative_file in negative_files:
        print("Scanning " + negative_file)
        sn, tm = scan_for_negatives(net, negative_file, args.threshold)
        if not sn:
            print("no negatives, moving on!")
            continue
        matching_patches = torch.stack(sn)
        patches_path = os.path.join(args.output_path, 'negatives_' + os.path.basename(negative_file) + '.pth')
        print("Writing output patches to:" + patches_path)
        torch.save(matching_patches, patches_path)

if __name__=="__main__":
    main()