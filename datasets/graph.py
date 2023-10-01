import logging, numpy as np


# Thanks to YAN Sijie for the released code on Github (https://github.com/yysijie/st-gcn)
class Graph:
    def __init__(self, dataset, max_hop=3, dilation=1):
        self.dataset = dataset.split('-')[0]
        self.max_hop = max_hop
        self.dilation = dilation

        # get edges
        self.num_node, self.edge, self.connect_joint, self.parts = self._get_edge()

        # get adjacency matrix
        self.A = self._get_adjacency()

    def __str__(self):
        return self.A

    def _get_edge(self):
        if self.dataset == 'mediapipe':
            # keypoints = {
            #     0: 'nose', 
            #     1: 'left_eye_inner', 2: 'left_eye', 3: 'left_eye_outer', 
            #     4: 'right_eye_inner', 5: 'right_eye', 6: 'right_eye_outer', 
            #     7: 'left_ear', 8: 'right_ear', 
            #     9: 'mouth_left', 10: 'mouth_right', 
            #     11: 'left_shoulder', 12: 'right_shoulder', 
            #     13: 'left_elbow', 14: 'right_elbow', 
            #     15: 'left_wrist', 16: 'right_wrist', 
            #     17: 'left_pinky', 18: 'right_pinky', 
            #     19: 'left_index', 20: 'right_index', 
            #     21: 'left_thumb', 22: 'right_thumb', 
            #     23: 'left_hip', 24: 'right_hip', 
            #     25: 'left_knee', 26: 'right_knee', 
            #     27: 'left_ankle', 28: 'right_ankle', 
            #     29: 'left_heel', 30: 'right_heel', 
            #     31: 'left_foot_index', 32: 'right_foot_index'
            # }
            num_node = 33
            self_link = [(i, i) for i in range(num_node)]
            neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 7), 
                             (0, 4), (4, 5), (5, 6), (6, 8), 
                             (0, 9), (0, 10), 
                             (0, 11), (0, 12), 
                             (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), 
                             (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), 
                             (11, 23), (12, 24), 
                             (23, 25), (25, 27), (27, 29), (29, 31), 
                             (24, 26), (26, 28), (28, 30), (30, 32)]
            self.edge = self_link + neighbor_link
            self.center = 0
            self.flip_idx = [0, 4, 5, 6, 1, 2, 3, 8, 7, 10, 9, 12, 11, 14, 13, 
                             16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25, 28, 
                             27, 30, 29, 32, 31] # look up table for mirrored body
            # root joint of every joints in sequence
            connect_joint = np.array([0, 0, 1, 2, 0, 4, 5, 3, 6, 0, 0, 0, 0, 11, 
                                      12, 13, 14, 15, 16, 15, 16, 15, 16, 11, 12,
                                      23, 24, 25, 26, 27, 28, 29, 30]) # for joint lengths
            parts = [
                np.array([11, 13, 15, 17, 19, 21]),       # left_arm
                np.array([12, 14, 16, 18, 20, 22]),      # right_arm
                np.array([23, 25, 27, 29, 31]),    # left_leg
                np.array([24, 26, 28, 30, 32]),    # right_leg
                np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),  # head
            ]
        elif self.dataset == 'mediapipe_noface': # DONE
            # keypoints = {
            #     0: 'nose', 
            #     1: 'left_shoulder', 2: 'right_shoulder', 
            #     3: 'left_elbow', 4: 'right_elbow', 
            #     5: 'left_wrist', 6: 'right_wrist', 
            #     7: 'left_pinky', 8: 'right_pinky', 
            #     9: 'left_index', 10: 'right_index', 
            #     11: 'left_thumb', 12: 'right_thumb', 
            #     13: 'left_hip', 14: 'right_hip', 
            #     15: 'left_knee', 16: 'right_knee', 
            #     17: 'left_ankle', 18: 'right_ankle', 
            #     19: 'left_heel', 20: 'right_heel', 
            #     21: 'left_foot_index', 22: 'right_foot_index'
            # }
            num_node = 23
            self_link = [(i, i) for i in range(num_node)]
            neighbor_link = [(0, 1), (1, 3), (3, 5), (5, 7), (5, 9), (5, 11), 
                             (0, 2), (2, 4), (4, 6), (6, 8), (6, 10), (6, 12), 
                             (1, 13), (13, 15), (15, 17), (17, 19), (19, 21), 
                             (2, 14), (14, 16), (16, 18), (18, 20), (20, 22)]
            self.edge = self_link + neighbor_link
            self.center = 0
            self.flip_idx = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 
                             16, 15, 18, 17, 20, 19, 22, 21] # look up table for mirrored body
            # root joint of every joints in sequence
            connect_joint = np.array([0, 0, 0, 1, 2, 3, 4, 5, 6, 5, 6, 5, 6, 1, 
                                      2, 13, 14, 15, 16, 17, 18, 19, 20]) # for joint lengths
            parts = [
                np.array([1, 3, 5, 7, 9, 11]),       # left_arm
                np.array([2, 4, 6, 8, 10, 12]),      # right_arm
                np.array([13, 15, 17, 19, 21]),    # left_leg
                np.array([14, 16, 18, 20, 22]),    # right_leg
                np.array([0]),  # head
            ]
        elif self.dataset == 'mediapipe_nohandfoot': # DONE
            # keypoints = {
            #     0: 'nose', 
            #     1: 'left_eye_inner', 2: 'left_eye', 3: 'left_eye_outer', 
            #     4: 'right_eye_inner', 5: 'right_eye', 6: 'right_eye_outer', 
            #     7: 'left_ear', 8: 'right_ear', 
            #     9: 'mouth_left', 10: 'mouth_right', 
            #     11: 'left_shoulder', 12: 'right_shoulder', 
            #     13: 'left_elbow', 14: 'right_elbow', 
            #     15: 'left_wrist', 16: 'right_wrist', 
            #     17: 'left_hip', 18: 'right_hip', 
            #     19: 'left_knee', 20: 'right_knee', 
            #     21: 'left_ankle', 22: 'right_ankle', 
            #     23: 'left_foot_index', 24: 'right_foot_index'
            # }
            num_node = 25
            self_link = [(i, i) for i in range(num_node)]
            neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 7), 
                             (0, 4), (4, 5), (5, 6), (6, 8), 
                             (0, 9), (0, 10), 
                             (0, 11), (11, 13), (13, 15), 
                             (0, 12), (12, 14), (14, 16), 
                             (11, 17), (17, 19), (19, 21), (21, 23), 
                             (12, 18), (18, 20), (20, 22), (22, 24)]
            self.edge = self_link + neighbor_link
            self.center = 0
            self.flip_idx = [0, 4, 5, 6, 1, 2, 3, 8, 7, 10, 9, 12, 11, 14, 13, 
                             16, 15, 18, 17, 20, 19, 22, 21, 24, 23] # look up table for mirrored body
            # root joint of every joints in sequence
            connect_joint = np.array([0, 0, 1, 2, 0, 4, 5, 3, 6, 0, 0, 0, 0, 11, 
                                      12, 13, 14, 11, 12, 17, 18, 19, 20, 21, 22]) # for joint lengths
            parts = [
                np.array([11, 13, 15]),       # left_arm
                np.array([12, 14, 16]),      # right_arm
                np.array([17, 19, 21, 23]),    # left_leg
                np.array([18, 20, 22, 34]),    # right_leg
                np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),  # head
            ]
        elif self.dataset == 'mediapipe_nofacehandfoot': # DONE
            # keypoints = {
            #     0: 'nose', 
            #     1: 'left_shoulder', 2: 'right_shoulder', 
            #     3: 'left_elbow', 4: 'right_elbow', 
            #     5: 'left_wrist', 6: 'right_wrist', 
            #     7: 'left_hip', 8: 'right_hip', 
            #     9: 'left_knee', 10: 'right_knee', 
            #     11: 'left_ankle', 12: 'right_ankle', 
            #     13: 'left_foot_index', 14: 'right_foot_index'
            # }
            num_node = 15
            self_link = [(i, i) for i in range(num_node)]
            neighbor_link = [(0, 1), (1, 3), (3, 5),  
                             (0, 2), (2, 4), (4, 6),  
                             (1, 7), (7, 9), (9, 11), (11, 13),  
                             (2, 8), (8, 10), (10, 12), (12, 14)]
            self.edge = self_link + neighbor_link
            self.center = 0
            self.flip_idx = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13] # look up table for mirrored body
            # root joint of every joints in sequence
            connect_joint = np.array([0, 0, 0, 1, 2, 3, 4, 1, 2, 7, 8, 9, 10, 11, 12]) # for joint lengths
            parts = [
                np.array([1, 3, 5]),       # left_arm
                np.array([2, 4, 6]),      # right_arm
                np.array([7, 9, 11, 13]),    # left_leg
                np.array([8, 10, 12, 14]),    # right_leg
                np.array([0]),  # head
            ]
        elif self.dataset == 'coco':
            # keypoints = {
            #     0: "nose",
            #     1: "left_eye",
            #     2: "right_eye",
            #     3: "left_ear",
            #     4: "right_ear",
            #     5: "left_shoulder",
            #     6: "right_shoulder",
            #     7: "left_elbow",
            #     8: "right_elbow",
            #     9: "left_wrist",
            #     10: "right_wrist",
            #     11: "left_hip",
            #     12: "right_hip",
            #     13: "left_knee",
            #     14: "right_knee",
            #     15: "left_ankle",
            #     16: "right_ankle"
            # }
            num_node = 17
            self_link = [(i, i) for i in range(num_node)]
            neighbor_link = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6),
                             (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
                             (11, 13), (13, 15), (12, 14), (14, 16)]
            self.edge = self_link + neighbor_link
            self.center = 0
            self.flip_idx = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
            connect_joint = np.array([0,0,0,1,2,0,0,5,6,7,8,5,6,11,12,13,14])
            parts = [
                np.array([5, 7, 9]),       # left_arm
                np.array([6, 8, 10]),      # right_arm
                np.array([11, 13, 15]),    # left_leg
                np.array([12, 14, 16]),    # right_leg
                np.array([0, 1, 2, 3, 4]),  # head
            ]

        elif self.dataset == 'oumvlp':
            # keypoints = {
            #     0: "nose",
            #     1: "neck"
            #     2: "Rshoulder"
            #     3: "Relbow"
            #     4: "Rwrist"
            #     5: "Lshoudler"
            #     6："Lelbow"
            #     7: "Lwrist"
            #     8: "Rhip"
            #     9: "Rknee"
            #     10: "Rankle"
            #     11: "Lhip"
            #     12: "Lknee"
            #     13: "Lankle"
            #     14: "Reye"
            #     15: "Leye"
            #     16: "Rear"
            #     17: "Lear"
            # }
            num_node = 18
            self_link = [(i, i) for i in range(num_node)]
            neighbor_link = [(0, 1), (0, 14), (0, 15), (14, 16), (15, 17),
                             (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
                             (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13)]
            self.edge = self_link + neighbor_link
            self.center = 1
            self.flip_idx = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16]
            connect_joint = np.array([1,1,1,2,3,1,5,6,2,8,9,5,11,12,0,0,14,15])
            parts = [
                np.array([5, 6, 7]),  # left_arm
                np.array([2, 3, 4]),  # right_arm
                np.array([11, 12, 13]),  # left_leg
                np.array([8, 9, 10]),  # right_leg
                np.array([0, 1, 14, 15, 16, 17]),  # head
            ]

        else:
            num_node, neighbor_link, connect_joint, parts = 0, [], [], []
            logging.info('')
            logging.error('Error: Do NOT exist this dataset: {}!'.format(self.dataset))
            raise ValueError()
        self_link = [(i, i) for i in range(num_node)]
        edge = self_link + neighbor_link
        return num_node, edge, connect_joint, parts

    def _get_hop_distance(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_adjacency(self):
        hop_dis = self._get_hop_distance()
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[hop_dis == hop] = 1
        normalize_adjacency = self._normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]
        return A

    @staticmethod
    def _normalize_digraph(A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD
