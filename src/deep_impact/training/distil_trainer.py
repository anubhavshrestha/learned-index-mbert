import torch

from .trainer import Trainer


class DistilMarginMSE:
    """MSE margin distillation loss from: Improving Efficient Neural Ranking Models with Cross-Architecture
    Knowledge Distillation
    link: https://arxiv.org/abs/2010.02666
    """

    def __init__(self):
        self.loss = torch.nn.MSELoss()

    def __call__(self, output, target):
        """
        Calculates the MSE loss between the teacher and student scores
        :param output: Shape (batch_size, n) with positive and negative predicted scores
        :param target: Shape (batch_size, n) with positive and negative teacher scores
        :return: MSE loss
        """
        student_positive_scores = output[:, 0]
        student_negative_scores = output[:, 1:]
        student_margin = student_positive_scores.unsqueeze(1) - student_negative_scores

        # Calculate margin for teacher
        teacher_positive_scores = target[:, 0]
        teacher_negative_scores = target[:, 1:]
        teacher_margin = teacher_positive_scores.unsqueeze(1) - teacher_negative_scores

        return self.loss(student_margin, teacher_margin)


class DistilKLLoss:
    """Distillation loss from: Distilling Dense Representations for Ranking using Tightly-Coupled Teachers
    link: https://arxiv.org/abs/2010.11386
    """

    def __init__(self):
        self.loss = torch.nn.KLDivLoss(reduction="none")

    def __call__(self, output, target):
        # Compute the KL divergence in float32 to avoid FP16 overflow/underflow issues.
        student_log_probs = torch.log_softmax(output.float(), dim=1)
        teacher_probs = torch.softmax(target.float(), dim=1)
        loss = self.loss(student_log_probs, teacher_probs).sum(dim=1).mean(dim=0)

        # When autocast runs in half precision, keep the loss in float32 for stability.
        if output.dtype == torch.float16:
            return loss
        return loss.to(output.dtype)


class DistilTrainer(Trainer):
    loss = DistilKLLoss()

    def evaluate_loss(self, outputs, batch):
        # distillation loss
        teacher_scores = batch['scores'].view(self.batch_size, -1).to(self.gpu_id)
        return self.loss(outputs, teacher_scores)
