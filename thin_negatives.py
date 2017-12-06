import torch
import glob
import os
import argparse

def subsample_negative_patches(negatives_path, size_to_sample=200000):
    "NOTE: subsamples approximately"
    negatives_files = glob.glob(os.path.join(negatives_path, '*.pth'))
    # count total number of patches
    total_sample_count = 0
    for path in negatives_files:
        samples = torch.load(path)
        total_sample_count += samples.size(0)
    print('total count is {}'.format(total_sample_count))

    # calculate odds of patch being in the subset we will sample, slightly increase the odds
    odds = size_to_sample*1.01 / total_sample_count

    sampled_data = []
    for path in negatives_files:
        #print("Starting with " + path)
        samples = torch.load(path)
        # generate selection: 0/1 according to odds per sample
        selections = torch.bernoulli(torch.Tensor([odds]).expand(samples.size(0)))
        selection_indices = selections.nonzero().squeeze()

        if len(selection_indices) == 0:
            #print("no patches selected")
            continue

        selected_patches = torch.index_select(samples, 0, selection_indices)
        sampled_data.append((path, selected_patches))
    return sampled_data

def main():
    main_arg_parser = argparse.ArgumentParser(description="options")
    main_arg_parser.add_argument("--input-path", help="path to negative mining output from mine_negatives.py", default="EX2_data/negative_mines/")
    main_arg_parser.add_argument("--output-file", help="path to write serialized negative patches", default="EX2_data/negative_mines_subset.pth")
    main_arg_parser.add_argument("--samples", help="output sample count", type=int, default=200000)
    args = main_arg_parser.parse_args()
    
    patches = subsample_negative_patches(args.input_path, args.samples)
    all_patches = torch.cat([p for f, p in patches])
    print("saving total of {} patches to {}".format(all_patches.size(0), args.output_file))
    torch.save(all_patches, args.output_file)

if __name__=="__main__":
    main()