import cv2
import seaborn as sns
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import copy

# functions to show an image
def draw_anomalous(input_img, fusion_img):
    input_img = input_img
    img_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    img_gray *= 255
    colors = sns.color_palette("hls", 7)
    for threshhold in range(1,8):
        ret, thresh = cv2.threshold(np.uint8(img_gray),(threshhold*36-1), 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, 3, 2)
        if contours:
            if threshhold == 1:
                hull = cv2.drawContours(fusion_img, contours, -1, colors[threshhold-2], 6)
            else:
                hull = cv2.drawContours(hull, contours, -1, colors[threshhold-2], 6)
        else :
            hull = fusion_img
    return hull

def pred_vs_gt(input_mask, gen_mask):
    RED = [1.0, 0.0, 0.0]
    BACKGROUND = [0.0, 0.46, 0.71]
    input_mask = np.where(input_mask!=0, 1, 0)
    gen_mask = np.where(gen_mask!=0, 1, 0)
    pred_error = np.where(input_mask!=gen_mask, RED, BACKGROUND)
    
    return pred_error

def MakeGrid(imgs, batch_size):
    img = np.transpose(make_grid(imgs, nrow=batch_size, padding=3, normalize=True).cpu(), (1, 2, 0))
    img = np.array(img, dtype='float32')

    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def show_images(input_imgs, input_masks, pred_masks=None,
                batch_size=5, savepath = ""):
    nrow = 3
    img_label = ["Images", "Ground truth\nMasks", "Ground truth\nFusions"]
    IMAGES = [input_imgs, input_masks]
    set_font = dict(fontsize=15,
                family='monospace',
                multialignment='left'
                )
    
    Grid_Image = []
    for imgs in IMAGES:
        img = MakeGrid(imgs, batch_size)
        Grid_Image.append(img)
    
    Grid_Image.append(draw_anomalous(Grid_Image[1], copy.deepcopy(Grid_Image[0])))
    
    if  pred_masks != None:
        nrow = 6
        img_label.extend(["Prediction\nMasks", "Prediction\nFusions", "Prediction\nvs\nGround truth"])
        Grid_Image.append(MakeGrid(pred_masks, batch_size))
        Grid_Image.append(draw_anomalous(Grid_Image[3], copy.deepcopy(Grid_Image[0])))
        Grid_Image.append(pred_vs_gt(Grid_Image[1], Grid_Image[3]))
        
    fig, ax = plt.subplots(nrows=nrow, ncols=1, figsize=(nrow*5, nrow*3))   
    
    for i in range(nrow):
        ax[i].set_title(img_label[i], fontdict=set_font,loc='left')
        ax[i].imshow(Grid_Image[i])
        ax[i].axis("off")
        
    if savepath != "":
        plt.savefig(savepath)
    
    fig.tight_layout()
    plt.show()

def during_loss(H_dict, title):
    
    sns.set_theme(style="darkgrid")

    plt.plot(H_dict['train_loss'])
    plt.plot(H_dict['valid_loss'])

    ax=plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(30))
    
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel('Epoch #') 
    plt.ylabel('Loss') 
    plt.legend(labels=["train_loss","test_loss"], loc="upper right")
    plt.show()

        