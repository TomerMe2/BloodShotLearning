import torch
from torchvision.transforms.functional import rotate


class ConsistencyLoss(torch.nn.Module):
    
    def __init__(self, number_of_consistency_check_per_data_point):
        super().__init__()
        self.number_of_consistency_check_per_data_point = number_of_consistency_check_per_data_point
        self.mse = torch.nn.MSELoss()
    
    def forward(self, x, embs, generate_emb_func):
        """
        x: batch of images
        embs: embeddings of x
        generate_embs_func: function that gets and image and returns the embedding of that image
        """
        
        x_for_consistent, embs_for_consistent = [], []
        for img, emb_of_img in zip(x, embs):
            
            for idx_of_consistency_check in range(self.number_of_consistency_check_per_data_point):
                img = rotate(img, torch.rand(1).item() * 360)
                x_for_consistent.append(img)
                embs_for_consistent.append(emb_of_img)
        
        x_for_consistent, embs_for_consistent = torch.stack(x_for_consistent), torch.stack(embs_for_consistent)
        embs_of_rotated = generate_emb_func(x_for_consistent)
        
        consistency_loss = self.mse(embs_for_consistent, embs_of_rotated)
        return consistency_loss