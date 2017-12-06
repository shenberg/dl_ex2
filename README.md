
Submitters:

Roee Shenberg, ID 200113041
Omer Goldman, ID 301803862
Raz Tamir, ID 026548768

Deployment notes:

All the paths necessary can be passed by command-line arguments, however, there are defaults that save all of this. For them to apply, the dir structure is assumed to be:
unzipped files in ./
Data folder called ./EX2_data containing:
./EX2_Data/aflw
./EX2_Data/fddb
./EX2_Data/VOC2007

In addition, create the following dirs:
./EX2_Data/fddb/out
./EX2_Data/negative_mines



Question 1
----------

To train a network:
    python3 q1.py --aflw-path <path to dir with aflw_12.t7> --voc-path <path to voc2007 root dir> [--cuda] [--visdom]


Net12: epoch 149, train loss: 0.03827351704239845, test loss: 0.04607975110411644, test accuracy: 0.982606873428332
threshold 0.11 to get recall >99% (0.9909576654336211). Resulting precision 0.9628594249201278


net12_precision_recall.png is the precision-recall curve
net12_loss.png is the training and test loss

Note: 2d dropout was used with p=0.1 on the convolutional layer for regularization, and there is still evidence of overfitting in the final loss

A large difference here is that the paper trained with 200k negative samples while we used ~25k, due to compute limitations.

Question 2
----------

To run on fddb, do as follows:
    python3 q2.py --checkpoint <path to serialized model from q1 e.g. q1_8db_dropout_150_epochs_threshold_0.11.pth.tar> --fddb-path <path to fddb> --output-path <dir to write fold-01-out.txt in> -t <threshold>

then from fddb dir, ./evaluation/runEvaluate.pl <output-path>

The graph with q1_8db_dropout_150_epochs_threshold_0.11.pth.tar shows that we get very high recall (>95%) with a great many false positives (~1k/image), so the threshold was adjusted to 0.25 to reduce the amount of windows to <500/image, lowering recall to %92.8 on fddb.

Attached are the discontinuous ROC graphs: 

DiscROC_net12_threshold_0.11.png
DiscROC_net12_threshold_0.25.png


Question 3
----------

To mine false negatives:
    python3 mine_negatives.py --checkpoint q1_somecheckpoint.pth.tar --voc-path <path to voc2007> --threshold <threshold found previously> --output-path <some existing dir>

To thin them out for computational feasibility in the given time constraints, the script thin_negatives.py was written, to sample a configurable amount out of the raw mined negatives and batch them together. this was used to reduce the 1.7 million patches gathered into 200k patches.

then q3.py was run, training 24-net.

Net24: epoch 399, train loss: 0.0214748, test loss: 0.02837808, test accuracy: 0.99253
threshold 0.004 to get recall >99% (0.9907407407407407). Resulting precision 0.5721925133689839

A more practical threshold turned out to be 0.001 in practical terms, discussed in q4.

The graphs are at:
24net_training_loss.png
24net_precision_recall.png


Question 4
----------

run q4.py, it's similar to q2, except now the flags are --net12 for 12net checkpoint ,--net24 for 24-net checkpoint, -t12 for threshold for 12-net, -t24 for threshold for 24-net

Attached is DiscROC_24net.png which shows ~82% recall with a total of 21550 FPs, meaning ~65 per image. Note that simply applying global NMS without filtering the boxes using 24-net reduced the ROC to 85%, so the loss from 24-net itself is very small. See DiscROC_global_nms_only.png for the graph.

In addition, the results with the 2nd threshold at 0.004 was also created at DiscROC_24net_threshold__0.004.png, with recall having dropped sharply to ~75% but less than 20 false positives per image.
