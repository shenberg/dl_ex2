import os
import argparse
import torch
import math
from torch.autograd import Variable
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from PIL import Image, ImageDraw
import models
import utils

def nms(boxes):
    if len(boxes) == 0:
        return []

    #xmin, ymin, xmax, ymax, score, sorted descending by score
    boxes = torch.stack(sorted(boxes, key=lambda x: x[4], reverse=True))

    pick = []
    x_min = boxes[:,0]
    y_min = boxes[:,1]
    x_max = boxes[:,2]
    y_max = boxes[:,3]

    area = (x_max-x_min)*(y_max-y_min)
    idxs = torch.arange(0, len(boxes)).long()
    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)
        if len(idxs) == 1:
            break
        xx1 = torch.max(x_min[i:i+1],x_min[idxs[1:]])
        yy1 = torch.max(y_min[i:i+1],y_min[idxs[1:]])
        xx2 = torch.min(x_max[i:i+1],x_max[idxs[1:]])
        yy2 = torch.min(y_max[i:i+1],y_max[idxs[1:]])

        w = torch.max(xx2-xx1, torch.zeros(1))
        h = torch.max(yy2-yy1, torch.zeros(1))

        # find relative overlap
        overlap = (w*h)/(area[idxs[1:]] + area[i] - w*h)
        no_overlap = overlap < 0.5
        # convert to indices in the list
        no_overlap_indexes = no_overlap.nonzero().squeeze() + 1

        if no_overlap_indexes.nelement() == 0: 
            # all remaining boxes overlap
            break

        idxs = idxs[no_overlap.nonzero().squeeze() + 1]
    
    return torch.stack([boxes[i] for i in pick])


def scan(net, image_tensor, threshold=0.1):
    res = net(Variable(image_tensor.unsqueeze(0)))
    #print(res.size(), (image_tensor.size(-2)+1)//2 - 5, (image_tensor.size(-1)+1)//2 - 5)
    #print(res.size(), image_tensor.size())
    # 2d indices of all pixels above threshold
    matches = (res.data > threshold)
    above_threshold_boxes = matches.squeeze(0).squeeze(0).nonzero()
    if len(above_threshold_boxes) == 0:
        return torch.Tensor()

    scores = res.data[matches]

    # convert coordinates to image space
    # compensate for halved image size
    matches = (above_threshold_boxes*2).float()

    upper_lefts = matches
    lower_rights = upper_lefts + torch.Tensor([12,12])
    xmin = upper_lefts[:,1:2]
    ymin = upper_lefts[:,0:1]
    xmax = lower_rights[:,1:2]
    ymax = lower_rights[:,0:1]
    boxes = torch.cat([xmin, ymin, xmax, ymax, scores], dim=1)
    return boxes


def get_image_patches(image_tensor, boxes):
    if len(boxes) == 0:
        return []
    patches = []
    for box in boxes.round().long():
        # index image tensor by y, then x (row, col)
        patches.append(image_tensor[:, box[1]:box[3], box[0]:box[2]])
    return patches


def scan_multiple_scales(net, image, scales, threshold):
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
        image_tensor = transform(image)

        # scan, nms on results
        results_for_scale = nms(scan(net, image_tensor, threshold).float())
        if len(results_for_scale) == 0:
            print("no results for scale {}".format(scale))
            continue
        patches = get_image_patches(image_tensor, results_for_scale)

        # now calculate exact scale for W,H since we rounded
        actual_size = image_tensor.size()
        # actual size is 3xHxW, reverse order here
        source_size = torch.Tensor([actual_size[-1], actual_size[-2]])
        
        real_scale = image_size_t / source_size

        # create [w,h,w,h,1] tensor for scaling the results (x1,y1,x2,y2,score)
        scale_tensor = torch.Tensor([real_scale[1], real_scale[0]]*2 + [1])

        total_matches.append((results_for_scale*scale_tensor, patches, scale))

    return total_matches


def fddb_scan(net, fddb_path, results_path, threshold):
    
    fddb_list = [line.strip() for line in open(os.path.join(fddb_path,'FDDB-folds/FDDB-fold-01.txt'))]

    with open(os.path.join(results_path, 'fold-01-out.txt'), 'w') as outfile:
        for fddb_item in fddb_list:
            print("Starting with " + fddb_item)
            image_path = os.path.join(fddb_path, 'images', fddb_item + '.jpg')
            image = default_loader(image_path)
            results = scan_multiple_scales(net, image, utils.scan_scales, threshold)

            result_count = sum([len(boxes) for boxes, *_ in results])
            outfile.write(str(fddb_item))
            outfile.write('\n')
            outfile.write(str(result_count))
            outfile.write('\n')
            outfile.write("\n".join("\n".join(utils.to_fddb_ellipses(boxes)) for boxes, *_ in results))
            outfile.write('\n')


def main():
    main_arg_parser = argparse.ArgumentParser(description="options")
    main_arg_parser.add_argument("--checkpoint", help="checkpoint file (from q1.py) path", default="q1_8db_dropout_150_epochs_threshold_0.11.pth.tar")
    main_arg_parser.add_argument("--fddb-path", help="path to FDDB root dir", default="EX2_data/fddb")
    main_arg_parser.add_argument("--output-path", help="path to write detection output files in (fold-01-out.txt)", default="EX2_data/fddb/out")
    # determined default by looking at precision-recall curve, choosing the most precision for >99% recall.
    main_arg_parser.add_argument("-t", "--threshold", help="positive cutoff", default=0.2, type=float)

    args = main_arg_parser.parse_args()

    net = models.Net12FCN()
    net.load_from_net12(args.checkpoint)
    net.eval()

    fddb_scan(net, args.fddb_path, args.output_path, args.threshold)


if __name__=="__main__":
    main()