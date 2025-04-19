import torch
import torch.nn as nn
import numpy as np

class MMCILoss(nn.Module):
    def __init__(self,alpha=1, beta=0.1, gamma=0.1): #之前是（0.1 1 0.1）
        """
        参数说明：
        alpha: 第一项的权重
        beta: 第二项（四次误差项）的权重
        gamma: 第三项（confidence项）的权重

        a(y_-y)^2+b(((y_+d_)-y)(y_-y))^2+c(d-d_)^2

        *b值可能大一些会好一些。要跟y_和y的均值是一个数量级
         a=c=1

        a(y_-y)^2+b(((y_+d_)-y)(y_-y))^2+c(d-d_)^2

        a(y_-y)^2+(b(y_-y)^2)((y_+d_-y)^2)+c(d-d_)^2

        所以a≈b(y_-y)^2≈c *这三项数量级上应该没有差距

        """
        super(MMCILoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward1(self, input, target, train_data = None):
        loss = torch.mean((input - target) ** 2)
        return loss


    def forward(self, input, target,train_data = None):

        """
        参数:
        outputs: 模型的输出，假设最后一个维度的第0个元素为预测电价 \overline{y_2}，
                 第1个元素为预测 confidence \overline{d}
        y: 真值电价，形状与预测电价相同；注意，真实 confidence 固定为 0
        """
        # preds = []
        # trues = []
        #
        # pred = input
        # true = target
        # #test_data = train_data
        # # preds.append(pred.detach().cpu().numpy())
        # # trues.append(true.detach().cpu().numpy())
        #
        # inverse_pred = train_data.inverse_transform(pred)
        # inverse_true = train_data.inverse_transform(true)
        l = 36
        y_pred = input[0][:, -l:, 0]
        y_1_pred = input[1][:, -l:, 0]
        y = target[:, -l:, 0]
        d_pred = input[0][:, -l:, 1]
        #d = target[:, -l:, 1]
        d = y-y_1_pred
        #d = y-y_pred #测试哪个效果好用哪个
        loss1 = torch.mean((y_1_pred - y) ** 2)
        loss2_1 = self.alpha * torch.mean((y_pred - y) ** 2)
        loss2_2 = self.beta * torch.mean(torch.abs(y_pred - y) * torch.abs(y_pred - y + d))
        loss2_3 = self.gamma * torch.mean((d - d_pred) ** 2)
        loss = loss1 + loss2_1 + loss2_2 + loss2_3
        return loss

        inverse_pred = input[0]
        inverse_true = target
        inverse_pred = inverse_pred.detach().cpu().numpy() #torch.Size([64,36,1])
        inverse_true = inverse_true.detach().cpu().numpy()

        inverse_pred_1 = input[1]
        inverse_pred_1 = inverse_pred_1.detach().cpu().numpy() #torch.Size([64,36,1])

        l = 36
        y_pred = inverse_pred[:, -l:, 0]
        y_1_pred = inverse_pred_1[:, -l:, 0]
        y = inverse_true[:, -l:, 0]
        d_pred = inverse_pred[:, -l:, 1]
        d = inverse_true[:, -l:, 1]

        # 分离出预测的电价和 confidence 缺少变量：
        # y_pred = input[..., 0]
        # d_pred = input[..., 1]
        # y = target[..., 0]
        # d = target[..., 1]
        # 总损失
        loss1 = np.mean((y_1_pred - y) ** 2)
        loss2_1 =  self.alpha * np.mean((y_pred - y) ** 2)
        loss2_2 = self.beta * np.mean(np.power((y_pred - y) ** 2,1/2) * np.power((y_pred - y + d) ** 2,1/2))

        loss2_3 = self.gamma * np.mean((d - d_pred) ** 2)
        loss2 = loss2_1 + loss2_2 + loss2_3
        loss = loss1 + loss2*0.1
        loss= torch.tensor(loss,device='cuda:0', requires_grad=True)
        print('=====================================')
        print(loss1,loss2,loss2_1,loss2_2,loss2_3,loss)
        return loss
class MMCISingleLossM1(nn.Module):
    def __init__(self,alpha=0.1, beta=1, gamma=0.1):
        """
        参数说明：
        alpha: 第一项的权重
        beta: 第二项（四次误差项）的权重
        gamma: 第三项（confidence项）的权重

        a(y_-y)^2+b(((y_+d_)-y)(y_-y))^2+c(d-d_)^2

        *b值可能大一些会好一些。要跟y_和y的均值是一个数量级
         a=c=1

        a(y_-y)^2+b(((y_+d_)-y)(y_-y))^2+c(d-d_)^2

        a(y_-y)^2+(b(y_-y)^2)((y_+d_-y)^2)+c(d-d_)^2

        所以a≈b(y_-y)^2≈c *这三项数量级上应该没有差距

        """
        super(MMCISingleLossM1, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward1(self, input, target, train_data = None):
        loss = torch.mean((input - target) ** 2)
        return loss


    def forward(self, input, target,train_data = None):

        """
        参数:
        outputs: 模型的输出，假设最后一个维度的第0个元素为预测电价 \overline{y_2}，
                 第1个元素为预测 confidence \overline{d}
        y: 真值电价，形状与预测电价相同；注意，真实 confidence 固定为 0
        """
        #return nn.functional.mse_loss(input, target)
        l = 36
        y_1_pred = input[:, -l:, 0]
        y = target[:, -l:, 0]
        loss1 = torch.mean((y_1_pred - y) ** 2)
        return loss1

        inverse_true = target
        inverse_true = inverse_true.detach().cpu().numpy()

        inverse_pred_1 = input
        inverse_pred_1 = inverse_pred_1.detach().cpu().numpy() #torch.Size([64,36,1])

        l = 36
        y_1_pred = inverse_pred_1[:, -l:, 0]
        y = inverse_true[:, -l:, 0]

        loss1 = torch.mean((y_1_pred - y) ** 2)
        loss= torch.tensor(loss1,device='cuda:0', requires_grad=True)
        print('=====================================')
        print(loss)
        return loss

class MMCISingleLossM2(nn.Module):
    def __init__(self,alpha=0.1, beta=1, gamma=0.1):
        """
        参数说明：
        alpha: 第一项的权重
        beta: 第二项（四次误差项）的权重
        gamma: 第三项（confidence项）的权重

        a(y_-y)^2+b(((y_+d_)-y)(y_-y))^2+c(d-d_)^2

        *b值可能大一些会好一些。要跟y_和y的均值是一个数量级
         a=c=1

        a(y_-y)^2+b(((y_+d_)-y)(y_-y))^2+c(d-d_)^2

        a(y_-y)^2+(b(y_-y)^2)((y_+d_-y)^2)+c(d-d_)^2

        所以a≈b(y_-y)^2≈c *这三项数量级上应该没有差距

        """
        super(MMCISingleLossM2, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward1(self, input, target, train_data = None):
        loss = torch.mean((input - target) ** 2)
        return loss


    def forward(self, input, target,train_data = None):

        """
        参数:
        outputs: 模型的输出，假设最后一个维度的第0个元素为预测电价 \overline{y_2}，
                 第1个元素为预测 confidence \overline{d}
        y: 真值电价，形状与预测电价相同；注意，真实 confidence 固定为 0
        """
        # preds = []
        # trues = []
        #
        # pred = input
        # true = target
        # #test_data = train_data
        # # preds.append(pred.detach().cpu().numpy())
        # # trues.append(true.detach().cpu().numpy())
        #
        # inverse_pred = train_data.inverse_transform(pred)
        # inverse_true = train_data.inverse_transform(true)
        l = 36
        y_pred = input[0][:, -l:, 0]
        y = target[:, -l:, 0]
        d_pred = input[0][:, -l:, 1]
        d = target[:, -l:, 1]
        loss2_1 = self.alpha * torch.mean((y_pred - y) ** 2)
        loss2_2 = self.beta * torch.mean(torch.abs(y_pred - y) * torch.abs(y_pred - y + d))
        loss2_3 = self.gamma * torch.mean((d - d_pred) ** 2)
        loss = loss2_1 + loss2_2 + loss2_3
        return loss

        inverse_pred = input[0]
        inverse_true = target
        inverse_pred = inverse_pred.detach().cpu().numpy() #torch.Size([64,36,1])
        inverse_true = inverse_true.detach().cpu().numpy()


        l = 36
        y_pred = inverse_pred[:, -l:, 0]
        y = inverse_true[:, -l:, 0]
        d_pred = inverse_pred[:, -l:, 1]
        d = inverse_true[:, -l:, 1]

        # 分离出预测的电价和 confidence 缺少变量：
        # y_pred = input[..., 0]
        # d_pred = input[..., 1]
        # y = target[..., 0]
        # d = target[..., 1]
        # 总损失
        loss2_1 = self.alpha * np.mean((y_pred - y) ** 2)
        loss2_2 = self.beta * np.mean(np.power((y_pred - y) ** 2,1/2) * np.power((y_pred - y + d) ** 2,1/2))

        loss2_3 = self.gamma * np.mean((d - d_pred) ** 2)
        loss = loss2_1 + loss2_2 + loss2_3
        loss= torch.tensor(loss,device='cuda:0', requires_grad=True)
        print('=====================================')
        print(loss2_1,loss2_2,loss2_3,loss)
        return loss
if __name__ == '__main__':

    # # 使用示例
    # loss_fn = MyMSELoss()
    # input_tensor = torch.tensor([1.0, 2.0, 3.0])
    # target_tensor = torch.tensor([1.5, 2.5, 3.5])
    # loss = loss_fn(input_tensor, target_tensor)
    # print("Loss:", loss.item())
    # print("Loss:", loss)
    # 构造测试数据
    # 输入：预测的电价和预测的 confidence
    input_tensor = torch.tensor([[3.5, 0.7],
                                 [2.1, -0.2],
                                 [4.2, 1.0],
                                 [3.8, 0.5]])

    # 目标：真实电价以及真实 confidence（固定为 0）
    target_tensor = torch.tensor([[3.0, 0.0],
                                  [2.0, 0.0],
                                  [4.0, 0.0],
                                  [3.9, 0.0]])

    # 初始化损失函数
    criterion = MMCILoss(alpha=1.0, beta=1.0, gamma=1.0)

    # 计算损失值
    loss_value = criterion(input_tensor, target_tensor)
    print("Loss:", loss_value.item())
