
Submitters:

Roee Shenberg, ID 200113041
You #1
You #2


Question 1
----------

To train a network:
    python3 q1.py --aflw-path <path to dir with aflw_12.t7> --voc-path <path to voc2007 root dir> [--cuda] [--visdom]


Net12: epoch 149, train loss: 0.03827351704239845, test loss: 0.04607975110411644, test accuracy: 0.982606873428332
threshold 0.11 to get recall >99% (0.9909576654336211). Resulting precision 0.9628594249201278


net12_precision_recall.png is the precision-recall curve
net12_loss.png is the training and test loss

Note: 2d dropout was used with p=0.1 on the convolutional layer for regularization, and there is still evidence of overfitting in the final loss

Question 2
----------

To run on fddb, do as follows:
    python3 q2.py --checkpoint <path to serialized model from q1 e.g. q1_8db_dropout_150_epochs_threshold_0.11.pth.tar> --fddb-path <path to fddb> --output-path <dir to write fold-01-out.txt in> -t <threshold>

then from fddb dir, ./evaluation/runEvaluate.pl <output-path>

The graph with q1_8db_dropout_150_epochs_threshold_0.11.pth.tar shows that we get very high recall (>95%) with a great many false positives (~1k/image), so the threshold was adjusted to 0.25 to reduce the amount of windows to ~500/image, lowering recall to %92.8 on fddb.

Attached are the discontinuous ROC graphs: 

DiscROC_net12_threshold_0.11.png
DiscROC_net12_threshold_0.25.png


Question 3
----------
To mine false negatives:
    python3 mine_negatives.py --checkpoint q1_somecheckpoint.pth.tar --voc-path <path to voc> --threshold <threshold found previously>

Question 4
----------

