import os
import argparse
import torch
from torch.autograd import Variable
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from PIL import Image, ImageDraw
import models
import q2



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

#TODO: threshold_24!!!
def scan_multiple_scales(net12, net24, image, scales, threshold_12=0.1, threshold_24=0.1):
    total_matches = []
    image_size_t = torch.Tensor([image.size[0], image.size[1]])
    # impose additional condition: destination size is even. helps deal with annoying rounding issues
    # that pop up because pytorch has no "SAME" rounding a la tensorflow
    dest_sizes = [(image_size_t*scale/2).round()*2 for scale in scales]
    
    # create scaled versions of the image, matches for them
    results_by_scale = []
    for scale, dest_size in zip(scales, dest_sizes):
        if any(dest_size < 12):
            print("image too small for given scale: {}, {}, {}".format(scale, image_size_t, dest_size))
            continue

        transform = transforms.Compose([transforms.Scale(dest_size.int()),
                                      transforms.ToTensor()])
        image_tensor = transform(image)
        # scan, nms on results
        matches = q2.nms(q2.scan(net12, image_tensor, threshold_12).float())

        # skip no matches for this scale 
        if len(matches) == 0:
            continue
        # append column of scale for each box
        matches = torch.cat([matches, torch.Tensor([scale]).expand(matches.size(0),1)], dim=1)
        results_by_scale.append((matches, image_tensor, scale))

    all_boxes = torch.cat([boxes for boxes, _, _ in results_by_scale])

    # apply global nms
    matches = q2.nms(all_boxes)

    filtered_matches = []
    # now run 24net on remaining matches, go by scale
    for scale, dest_size in zip(scales, dest_sizes):
        if any(dest_size < 12):
            continue

        # take all rows where row's 6th column (scale) matches current scale
        matching_rows = (matches[:,5] == scale).nonzero().squeeze()
        if len(matching_rows) == 0:
            print("no matches for scale {}".format(scale))
            continue

        matches_for_scale = torch.index_select(matches, 0, matching_rows)

        double_transform = transforms.Compose([transforms.Scale(dest_size.int()*2),
                              transforms.ToTensor()])
        double_image_tensor = double_transform(image)

        patches = get_image_patches(double_image_tensor, matches_for_scale.round()*2)

        # run 24net on all patches
        scores = net24(Variable(torch.stack(patches)))

        # now calculate exact scale for W,H since we rounded
        actual_size = image_tensor.size()
        # actual size is 3xHxW, reverse order here
        source_size = torch.Tensor([actual_size[-1], actual_size[-2]])
        
        real_scale = image_size_t / source_size

        # create [w,h,w,h,1] tensor for scaling the results (x1,y1,x2,y2,score)
        scale_tensor = torch.Tensor([real_scale[1], real_scale[0]]*2 + [1])

        total_matches.append((matches_for_scale[:,:5]*scale_tensor, patches, scores, scale))

    return total_matches


def to_fddb_ellipses(boxes):
    centers = (boxes[:,0:2] + boxes[:,2:4]) * 0.5
    dims = boxes[:,2:4] - boxes[:,0:2]
    axes = dims * torch.Tensor([1.13, 1.18])
    # major axis, minor axis, angle, center x, center y, score
    ellipses = torch.cat([axes, torch.zeros(boxes.size(0), 1), centers, boxes[:,4]], dim=1)
    # stringify
    return ["{} {} {} {} {} {}".format(*row) for row in ellipses]

# TODO
def fddb_scan(net12, net24, fddb_path, results_path):
    
    fddb_list = [line.strip() for line in open(os.path.join(fddb_path,'FDDB-folds/FDDB-fold-01.txt'))]

    with open(os.path.join(results_path, 'fold-01-out.txt'), 'w') as outfile:
        for fddb_item in fddb_list:
            print("Starting with " + fddb_item)
            image_path = os.path.join(fddb_path, 'images', fddb_item + '.jpg')
            image = default_loader(image_path)
            #results = scan_multiple_scales(net12, net24, image, [0.2, 0.16, 0.13, 0.1])
            results = scan_multiple_scales(net12, net24, image, [0.4, 0.32, 0.3, 0.25, 0.22, 0.2,0.07,0.09, 0.16, 0.13, 0.1, 0.05])

            result_count = sum([len(boxes) for boxes, *_ in results])
            outfile.write(str(fddb_item))
            outfile.write('\n')
            outfile.write(str(result_count))
            outfile.write('\n')
            outfile.write("\n".join("\n".join(to_fddb_ellipses(boxes)) for boxes, *_ in results))
            outfile.write('\n')

threshold = 0.1 # determined by looking at precision-recall curve, choosing the most precision for >99% recall.

def main():
    main_arg_parser = argparse.ArgumentParser(description="options")
    main_arg_parser.add_argument("--net12", help="checkpoint file for 12net (from q1.py) path", default="q2_start_3db_no_regularization.pth.tar")
    main_arg_parser.add_argument("--net24", help="checkpoint file for 12net (from q3.py) path", default="q3_checkpoint_400_epochs_batchnorm2d_lr_schedule_100_200_300.pth.tar")
    main_arg_parser.add_argument("--fddb-path", help="path to FDDB root dir", default="EX2_data/fddb")
    main_arg_parser.add_argument("--output-path", help="path to write detection output files in (fold-01-out.txt)", default="EX2_data/fddb/out")
    args = main_arg_parser.parse_args()

    net12 = models.Net12FCN()
    net12.load_from_net12(args.net12)

    net24 = models.Net24()
    net24.load_state_dict(torch.load(args.net24)['state_dict'])

    fddb_scan(net12, net24, args.fddb_path, args.output_path)


if __name__=="__main__":
    main()