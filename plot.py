
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from matplotlib import pyplot as plt



def plot_confusion_matrix(y_preds, y_true, classes):
    cm = confusion_matrix(y_true, y_preds)
    fig, ax = plt.subplots(figsize=(25, 25))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels= classes)
    disp.plot( ax=ax, colorbar=False)
    fig.suptitle('BI-LSTM Confusion matrix', fontsize=27)
    #plt.title("BI-LSTM Confusion matrix")
    plt.show()