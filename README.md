
Submitters:

Roee Shenberg, ID 200113041
You #1
You #2


Question 1
----------

To train a network:
    python3 q1.py --aflw-path <path to dir with aflw_12.t7> --voc-path <path to voc2007 root dir> [--cuda] [--visdom]


train loss: 0.03446, test loss: 0.04352, test classification accuracy: 0.984

net12_precision.png is the precision-recall curve
net12_loss.png is the training and test loss

Note: 2d dropout was used with p=0.1 on the convolutional layer for regularization, and there is still evidence of overfitting in the final loss

#TODO: explain threshold from recall-precision prints

Question 2
----------

To run on fddb, do as follows:
    python3 q2.py --checkpoint <path to serialized model from q1 e.g. q1_checkpoint.pth.tar> --fddb-path <path to fddb> --output-path <dir to write fold-01-out.txt in>

then from fddb dir, ./evaluation/runEvaluate.pl <output-path>

DiscROC_net12.png is the resulting graph (--threshold 0.2 --)
NOTE: DiscROC curve for reporting all windows as matches for the set of scales 1.18**(-2k+1) for k in {1..10} returned poor results, implying the conversion to fddb matches is somehow flawed. The curve is attached as DistROC_dense_scan.png and shows only 93% recall for millions(!) of false positives.

Question 3
----------
To mine false negatives:
    python3 mine_negatives.py --checkpoint q1_somecheckpoint.pth.tar --voc-path <path to voc> --threshold <threshold found previously>

Question 4
----------

