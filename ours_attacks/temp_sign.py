import torch

class temp_sign_comparison():
    def __init__(self, video, epsilon, grad, c, targeted):
        self.video = video
        self.epsilon = epsilon
        self.grad = grad
        self.targeted = targeted
        self.c = c
        
    def forward(self):
        self.sing_grad = self.grad.sign()
        self.sing_grad = self.sing_grad.squeeze(0).permute(1,0,2,3)
        
        self.epsilon *= self.c
        cnt = 0
        
        for frame_index in range(0, self.sing_grad.shape[0], 2):

            tf = torch.eq(self.sing_grad[frame_index], self.sing_grad[frame_index + 1])

            self.sing_grad[frame_index: frame_index + 1].squeeze(0)[tf] = 0
            self.sing_grad[frame_index + 1: frame_index + 2].squeeze(0)[tf] = 0

            cnt += 2 * torch.sum(tf == False)

        total = self.sing_grad.flatten().shape[0]

        self.sing_grad = self.sing_grad.unsqueeze(0)
        self.sing_grad = self.sing_grad.permute(0, 2, 1, 3, 4)

        if self.targeted:
            adv_video = self.video - ((total/cnt) * self.epsilon) *  self.sing_grad
            adv_video = torch.clamp(adv_video, 0, 1)
        else:
            adv_video = self.video + ((total/cnt) * self.epsilon) *  self.sing_grad
            adv_video = torch.clamp(adv_video, 0, 1)

        return adv_video, self.c