import torch

class Conv2d_batchnorm(torch.nn.Module):

    def __init__(self, num_in_filters, num_out_filters, kernel_size, stride=1, is_shortcut=False):
        super().__init__()
        self.is_shortcut = is_shortcut
        self.conv1 = torch.nn.Conv2d(in_channels=num_in_filters, out_channels=num_out_filters,
                                     kernel_size=kernel_size, stride=stride, padding='same')
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)

        if self.is_shortcut:
            return x
        else:
            return torch.nn.functional.relu(x)


class MultiResblock(torch.nn.Module):

    def __init__(self, num_in_channels, num_filters, alpha=1.67):

        super().__init__()
        self.alpha = alpha
        self.W = num_filters * alpha

        filt_cnt_3x3 = int(self.W*0.167)
        filt_cnt_5x5 = int(self.W*0.333)
        filt_cnt_7x7 = int(self.W*0.5)
        num_out_filters = filt_cnt_3x3 + filt_cnt_5x5 + filt_cnt_7x7

        self.shortcut = Conv2d_batchnorm(
            num_in_channels, num_out_filters, kernel_size=1, is_shortcut=True)

        self.conv_3x3 = Conv2d_batchnorm(
            num_in_channels, filt_cnt_3x3, kernel_size=3)

        self.conv_5x5 = Conv2d_batchnorm(
            filt_cnt_3x3, filt_cnt_5x5, kernel_size=3)

        self.conv_7x7 = Conv2d_batchnorm(
            filt_cnt_5x5, filt_cnt_7x7, kernel_size=3)

        self.batch_norm1 = torch.nn.BatchNorm2d(num_out_filters)
        self.batch_norm2 = torch.nn.BatchNorm2d(num_out_filters)

    def forward(self, x):

        shortcut = self.shortcut(x)

        a = self.conv_3x3(x)
        b = self.conv_5x5(a)
        c = self.conv_7x7(b)

        x = torch.cat([a, b, c], axis=1)
        x = self.batch_norm1(x)

        x = x + shortcut
        x = self.batch_norm2(x)
        x = torch.nn.functional.relu(x)

        return x


class ResPath(torch.nn.Module):

    def __init__(self, num_in_filters, num_out_filters, ResPath_length):

        super().__init__()

        self.ResPath_length = ResPath_length
        self.shortcuts = torch.nn.ModuleList([])
        self.convs = torch.nn.ModuleList([])
        self.bns = torch.nn.ModuleList([])

        for i in range(self.ResPath_length):
            if(i == 0):
                self.shortcuts.append(Conv2d_batchnorm(
                    num_in_filters, num_out_filters, kernel_size=1, is_shortcut=True))
                self.convs.append(Conv2d_batchnorm(
                    num_in_filters, num_out_filters, kernel_size=3))
            else:
                self.shortcuts.append(Conv2d_batchnorm(
                    num_out_filters, num_out_filters, kernel_size=1, is_shortcut=True))
                self.convs.append(Conv2d_batchnorm(
                    num_out_filters, num_out_filters, kernel_size=3))

            self.bns.append(torch.nn.BatchNorm2d(num_out_filters))

    def forward(self, x):

        for i in range(self.ResPath_length):

            shortcut = self.shortcuts[i](x)

            x = self.convs[i](x)
            x = self.bns[i](x)
            x = torch.nn.functional.relu(x)

            x = x + shortcut
            x = self.bns[i](x)
            x = torch.nn.functional.relu(x)

        return x

class MultiResUnetDown(torch.nn.Module):

    def __init__(self, in_channels, out_channels, in_filters, ResPath_length):
        super().__init__()

        self.MultiResblock = MultiResblock(in_channels, out_channels)
        self.pool = torch.nn.MaxPool2d(2)
        self.ResPath = ResPath(in_filters, out_channels, ResPath_length=ResPath_length)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_multiRes = self.MultiResblock(x)
        x_pool = self.pool(x_multiRes)
        x_multiRes = self.ResPath(x_multiRes)
        
        return x_multiRes, x_pool
    

class MultiResUnet(torch.nn.Module):

    def __init__(self, input_channels, num_classes, alpha=1.67):
        super().__init__()

        self.alpha = alpha

        # Encoder
        self.in_filters1 = int(32*self.alpha*0.167) + int(32*self.alpha*0.333) + int(32*self.alpha * 0.5)
        self.in_filters2 = int(32*2*self.alpha*0.167) + int(32*2*self.alpha*0.333) + int(32*2*self.alpha * 0.5)
        self.in_filters3 = int(32*4*self.alpha*0.167) + int(32*4*self.alpha*0.333) + int(32*4*self.alpha * 0.5)            
        self.in_filters4 = int(32*8*self.alpha*0.167) + int(32*8*self.alpha*0.333) + int(32*8*self.alpha * 0.5)
                        
        self.Downsample1 = MultiResUnetDown(in_channels = input_channels, out_channels = 32, in_filters = self.in_filters1, ResPath_length = 4)
        self.Downsample2 = MultiResUnetDown(in_channels = self.in_filters1, out_channels = 32*2, in_filters = self.in_filters2, ResPath_length = 3)
        self.Downsample3 = MultiResUnetDown(in_channels = self.in_filters2, out_channels = 32*4, in_filters = self.in_filters3, ResPath_length = 2)
        self.Downsample4 = MultiResUnetDown(in_channels = self.in_filters3, out_channels = 32*8, in_filters = self.in_filters4, ResPath_length = 1)

        #bridge
        self.in_filters5 = int(32*16*self.alpha*0.167) + int(32*16*self.alpha*0.333) + int(32*16*self.alpha * 0.5)
        self.bridge = MultiResblock(self.in_filters4, 32*16)
        
        # Decoder
        self.in_filters6 = int(32*8*self.alpha*0.167) + int(32*8*self.alpha*0.333) + int(32*8*self.alpha * 0.5)
        self.in_filters7 = int(32*4*self.alpha*0.167) + int(32*4*self.alpha*0.333) + int(32*4*self.alpha * 0.5)
        self.in_filters8 = int(32*2*self.alpha*0.167) + int(32*2*self.alpha*0.333) + int(32*2*self.alpha * 0.5)
        self.in_filters9 = int(32*self.alpha*0.167) + int(32*self.alpha*0.333) + int(32*self.alpha * 0.5)
                        
        self.concat_filters1 = 32 * 8 * 2
        self.concat_filters2 = 32 * 4 * 2
        self.concat_filters3 = 32 * 2 * 2
        self.concat_filters4 = 32 * 2
        
        self.in_filters6 = int(32*8*self.alpha*0.167) + int(32*8*self.alpha*0.333) + int(32*8*self.alpha* 0.5)
        self.in_filters7 = int(32*4*self.alpha*0.167) + int(32*4*self.alpha*0.333) + int(32*4*self.alpha* 0.5)
        self.in_filters8 = int(32*2*self.alpha*0.167) + int(32*2*self.alpha*0.333) + int(32*2*self.alpha* 0.5)
        self.in_filters9 = int(32*self.alpha*0.167) + int(32*self.alpha*0.333) + int(32*self.alpha* 0.5)
        
        
        self.Upsample6 = torch.nn.ConvTranspose2d(self.in_filters5,32*8,kernel_size=(2,2),stride=(2,2))  
        self.MultiResblock6 = MultiResblock(self.concat_filters1,32*8)
        
        self.Upsample7 = torch.nn.ConvTranspose2d(self.in_filters6,32*4,kernel_size=(2,2),stride=(2,2))  
        self.MultiResblock7 = MultiResblock(self.concat_filters2,32*4)
	
        self.Upsample8 = torch.nn.ConvTranspose2d(self.in_filters7,32*2,kernel_size=(2,2),stride=(2,2))
        self.MultiResblock8 = MultiResblock(self.concat_filters3,32*2)
	
        self.Upsample9 = torch.nn.ConvTranspose2d(self.in_filters8,32,kernel_size=(2,2),stride=(2,2))
        self.MultiResblock9 = MultiResblock(self.concat_filters4,32)
        
        self.conv_final = Conv2d_batchnorm(
            self.in_filters9, num_classes+1, kernel_size=1, is_shortcut=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:


        x_multiRes1, x_pool1 = self.Downsample1(x)
        x_multiRes2, x_pool2 = self.Downsample2(x_pool1)
        x_multiRes3, x_pool3 = self.Downsample3(x_pool2)
        x_multiRes4, x_pool4 = self.Downsample4(x_pool3)

        x_multiRes5 = self.bridge(x_pool4)

        up6 = torch.cat([self.Upsample6(x_multiRes5), x_multiRes4], axis=1)
        x_multiRes6 = self.MultiResblock6(up6)

        up7 = torch.cat([self.Upsample7(x_multiRes6), x_multiRes3], axis=1)
        x_multiRes7 = self.MultiResblock7(up7)

        up8 = torch.cat([self.Upsample8(x_multiRes7), x_multiRes2], axis=1)
        x_multiRes8 = self.MultiResblock8(up8)

        up9 = torch.cat([self.Upsample9(x_multiRes8), x_multiRes1], axis=1)
        x_multiRes9 = self.MultiResblock9(up9)

        out = self.conv_final(x_multiRes9)

        return out
