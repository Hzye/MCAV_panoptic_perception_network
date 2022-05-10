import torch
import torch.nn as nn

class Yolo_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        # lambda constants
        self.lambda_class = 1
        self.lambda_noobj = 5
        self.lambda_box = 5
        self.lambda_obj = 1


    def forward(self, prediction, label):
        """
        Computes difference between prediction and label.
        
        Input:
        =prediction=    Tensor of all prediction arrays of size (n_batches, 10647, 5+n_classes).
        =label=         Tensor of all label arryays of size (n_batches, 10647, 5+n_classes).
        
        Output:
        =loss=          Total loss computed for this batch.
        """
        batch_size = prediction.shape[0]

        # I^obj_i in paper
        # mask for actual object in grid
        obj_i = (label[:,:,4] == 1) # size (batch_size, 10647)

        # I^obj_ij in paper
        # mask for when there IS obj in label AND box has highest conf score
        confs = prediction[:,:,4].reshape(batch_size, -1, 3) # reshape to easily find argmax(box1,box2,box3)
        highest_conf = torch.argmax(confs.reshape(batch_size, -1, 3), axis=2)
        mask = torch.arange(confs.reshape(batch_size, -1, 3).size(2)).reshape(1, 1, -1) == highest_conf.unsqueeze(2) # create T/F mask
        mask = mask.reshape(batch_size, -1) # reshape back to (batch_size, 10647)
        # now AND with (there is object) mask
        obj_ij = mask*obj_i # size (batch_size, 10647)

        # I^noobj_ij in paper
        noobj_i = (label[:,:,4] == 0) # true if there are no objects, size (batch_size, 10647)
        noobj_ij = mask*noobj_i # size (batch_size, 10647)

        ## box loss
        # use generic square diff loss (mse)
        x_mse = torch.square(prediction[:,:,0] - label[:,:,0])
        y_mse = torch.square(prediction[:,:,1] - label[:,:,1])
        bbox_centre_mse = x_mse + y_mse
        bbox_centre_loss = torch.sum(obj_ij*bbox_centre_mse)

        w_mse = torch.square(torch.sqrt(prediction[:,:,2]) - torch.sqrt(label[:,:,2]))
        h_mse = torch.square(torch.sqrt(prediction[:,:,3]) - torch.sqrt(label[:,:,3]))
        bbox_dims_mse = w_mse + h_mse
        bbox_dims_loss = torch.sum(obj_ij*bbox_dims_mse)

        bbox_loss = (1/batch_size)*(bbox_centre_loss + bbox_dims_loss)

        ## object loss
        # use binary cross entropy loss
        t1 = label[:,:,4]*torch.log(prediction[:,:,4])
        t2 = (1 - label[:,:,4])*torch.log(1 - prediction[:,:,4])
        obj_bce = t1 + t2
        obj_loss = -(1/batch_size)*torch.sum(obj_ij*obj_bce)

        ## no object loss
        noobj_loss = -(1/batch_size)*torch.sum(noobj_ij*obj_bce)

        ## class loss
        # use cross entropy loss
        class_loss = -torch.sum(obj_i*torch.sum(label[:,:,-12:]*torch.log(prediction[:,:,-12:]), axis=2))

        ## combine all losses
        loss = self.lambda_box*bbox_loss + self.lambda_obj*obj_loss + self.lambda_noobj*noobj_loss + self.lambda_class*class_loss

        return loss