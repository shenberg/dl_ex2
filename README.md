
Submitters:

Roee Shenberg, ID 200113041
You #1
You #2


Question 1
----------

To train a network:
    python3 q1.py --aflw-path <path to dir with aflw_12.t7> --voc-path <path to voc root dir> [--cuda] [--visdom]


train loss: 0.03446, test loss: 0.04352, test classification accuracy: 0.984

net12_precision.png is the precision-recall curve
net12_loss.png is the training and test loss

Note: 2d dropout was used with p=0.1 on the convolutional layer for regularization, and there is still evidence of overfitting in the final loss

#TODO: explain threshold from recall-precision prints

Question 2
----------

To run on fddb, do as follows:
    python3 q2.py --checkpoint <path to serialized model from q1 e.g. q1_checkpoint.pth.tar> --fddb-path <path to fddb> --output-path <dir to write fold-01-out.txt in>

then from fddb dir, ./eval

#TODO: run python q2.py after final scales found with -t 0.2 --checkpoint <q1_batchnorm> to compare with q2_3db network


Question 3
----------


Question 4
----------

