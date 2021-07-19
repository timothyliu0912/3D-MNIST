from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import tensorflow as tf
def plot_conf(pred_class,y_test):
    y_tar = tf.math.argmax(y_test,1)
    cm = confusion_matrix(pred_class,y_tar)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot() 