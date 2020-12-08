import unittest
import numpy as np
import torch


def get_batch_idx(batch, actions):
    batch_num_nodes = torch.bincount(batch)
    batch_size = batch_num_nodes.shape[0]
    cumsum = torch.cumsum(batch_num_nodes, dim=0) - batch_num_nodes[0]

    a_first = cumsum + actions[:, 0]
    a_second = cumsum + actions[:, 1]
    return a_first, a_second, batch_num_nodes


class TestGetBatchIdx(unittest.TestCase):

    def test_1(self):
        """Batches of graphs with the same number of nodes and a random action array"""

        batch = torch.from_numpy(np.repeat(np.arange(2000), 20))
        actions = torch.from_numpy(np.tile([1, 2], 2000).reshape(2000, 2))

        a_first, a_second, _ = get_batch_idx(batch, actions)
        correct_first, correct_second = torch.from_numpy(np.arange(2000) * 20) + np.ones(2000), \
                                        torch.from_numpy(np.arange(2000) * 20) + 2*np.ones(2000)
        self.assertTrue(torch.all(torch.eq(a_first, correct_first)))
        self.assertTrue(torch.all(torch.eq(a_second, correct_second)))

    def test_2(self):
        """Batches of graphs with the same number of nodes and actions from the last node in the graph"""

        batch = torch.from_numpy(np.repeat(np.arange(2000), 20))
        actions = torch.from_numpy(np.tile([19, 19], 2000).reshape(2000, 2))

        a_first, a_second, _ = get_batch_idx(batch, actions)
        correct_first, correct_second = torch.from_numpy(np.arange(2000) * 20) + np.ones(2000)*19, \
                                        torch.from_numpy(np.arange(2000) * 20) + np.ones(2000)*19
        self.assertTrue(torch.all(torch.eq(a_first, correct_first)))
        self.assertTrue(torch.all(torch.eq(a_second, correct_second)))


if __name__ == '__main__':
    unittest.main()