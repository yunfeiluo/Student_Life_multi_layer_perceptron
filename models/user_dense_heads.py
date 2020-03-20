import warnings
import torch
import torch.nn as nn

from src.definitions import LOW_MODEL_CAPACITY_WARNING

# ---------------------------------------------------------------- 
# Independent Study 496, Student Stree Prediction
#
# Class_name: GroupDenseHead
# Functionality: define MLP of each group of students
# Author: Yunfei Luo
# Start date: EST Feb.22th.2020
# Last update: EST Mar.20th.2020
# ----------------------------------------------------------------

class GroupDenseHead(nn.Module):
    def __init__(self, groups: dict, input_size, hidden_size, num_classes, dropout=0, ordinal_regression_head=False):
        """
        This model has a dense layer for each group of students. This is used for MultiTask learning.

        @param groups: dictionary of groups of student, map: student_ids -> group_ids
        The ids of group and student much be strings.
        @param input_size: Input size of each dense layer.
        @param hidden_size: Hidden size of the dense layer.
        """
        super(GroupDenseHead, self).__init__()
        self.is_cuda_avail = True if torch.cuda.device_count() > 0 else False
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.groups = groups # map: student -> group

        group_nodes = set()
        for student in groups:
            group_nodes.add(groups[student])

        # Layer initialization.
        if self.input_size > self.hidden_size:
            warnings.warn(LOW_MODEL_CAPACITY_WARNING)
        dense_layer = dict()

        # make a dense layer for each group
        for group in group_nodes:
            sequential_liner = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(self.hidden_size, self.num_classes))
            
            dense_layer[group] = sequential_liner

        self.student_dense_layer = nn.ModuleDict(dense_layer)

    def forward(self, user, input_data):
        return self.student_dense_layer[self.groups[user]](input_data)
