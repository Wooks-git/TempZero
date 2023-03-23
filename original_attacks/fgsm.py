import torch
import torch.nn as nn


class FGSM():
    def __init__(self, video, epsilon, grad, targeted):
        self.video = video
        self.epsilon = epsilon
        self.grad = grad
        self.targeted = targeted
        
    def forward(self):
        self.sing_grad = self.grad.sign()
        
        if self.targeted:
            adv_video = self.video - self.epsilon * self.sing_grad
            adv_video = torch.clamp(adv_video, 0,1)
        else:
            adv_video = self.video + self.epsilon * self.sing_grad
            adv_video = torch.clamp(adv_video, 0,1)
        
        return adv_video