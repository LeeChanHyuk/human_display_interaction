import torch
import os
import sys
import yaml
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from template.trainer.architecture import action_transformer
from main_utils import preprocessing

def build_model(num_classes=-1):
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(base_path, "template/conf/architecture/action_transformer.yaml")) as f:
        list_doc = yaml.load(f.read(), Loader=yaml.FullLoader)
        order = list_doc['mode']
        architecture = action_transformer.ActionTransformer3(
            ntoken=list_doc['ntoken'],
            nhead=list_doc['nhead'][order],
            dropout=list_doc['dropout'][order],
            mlp_size=list_doc['mlp_size'][order],
            classes=list_doc['classes'],
            nlayers=list_doc['nlayers'][order],
            sequence_length=list_doc['sequence_length'],
            alpha = list_doc['alpha'],
            n_hid = list_doc['gat_output_dim'][order],
            softmax_dim=list_doc['softmax_dim']
        )
        model = architecture.to('cuda', non_blocking=True)
        checkpoint_name = os.path.join(base_path, 'output/head_pose_based_best.pth.tar')
        #model = DDP(model, device_ids=[0], output_device=0, find_unused_parameters=True)
        return model, checkpoint_name
    return None

def load_model_for_inference(rank):
    model, checkpoint_name = build_model()
    # For using DDP
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    base_path = os.getcwd()
    checkpoint_path = os.path.join(base_path, checkpoint_name)
    # Load state_dict
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def inference(pose_sequence):
    # action_vote variable is used for classifying which action is most detected
    action_vote = np.zeros((18), dtype=np.uint8)

    # Action recognition threshold
    threshold = 0.90

    # numpy array -> torch.Tensor
    pose_sequence = torch.Tensor(pose_sequence).to(device='cuda')

    # Prediction
    y_pred = model(pose_sequence)

    # using activation function (softmax)
    y_pred = torch.softmax(y_pred, dim=1)

	# torch.Tensor -> numpy.array
    y_pred = y_pred.detach().cpu().numpy()

    # Maximum probability of action
    probability = np.max(y_pred)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = np.around(y_pred)
    y_pred = y_pred[0]

    # ?????? ????????? ??? ?????? ?????? ??????. (?????? frame??? ?????? ??????????????????)
    for i in range(results.size-1):
        results[i] = results[i+1]
        action_vote[results[i]] += 1
    results[-1] = y_pred
    action_vote[results[-1]] += 1
    max_voted_action_val = np.max(action_vote)
    max_voted_action_class = np.argmax(action_vote)
    
    # ?????? ?????? ????????? list??? ?????????.
    actions = [ 'nolooking', 'yaw-', 'yaw+', 'pitch-', 'pitch+', 'roll-', 'roll+', 'left', 'left-up', 'up',
    'right-up', 'right', 'right-down', 'down', 'left-down', 'zoom-in', 'zoom-out','standard']
    # if the voted action class is over than 10 and the class is standard
    if probability > threshold and (max_voted_action_val > 10 and y_pred == max_voted_action_class) and max_voted_action_class == 0:
        state = 'standard'
    elif probability > threshold and (max_voted_action_val > 10 and y_pred == max_voted_action_class):
        state = actions[max_voted_action_class]
    else:
        state = 'standard'
    return state

def action_recognition_func(human_info, fps = 20):
    # Convert the list to numpy array
    center_eyes = np.array(human_info.center_eyes[-2*fps:])
    center_mouths = np.array(human_info.center_mouths[-2*fps:])
    left_shoulders = np.array(human_info.left_shoulders[-2*fps:])
    right_shoulders = np.array(human_info.right_shoulders[-2*fps:])
    center_stomachs = np.array(human_info.center_stomachs[-2*fps:])
    head_poses = np.array(human_info.head_poses[-2*fps:])
    network_input = np.array([center_eyes, center_mouths, left_shoulders, right_shoulders, center_stomachs, head_poses])

    # ??? 25?????? features
    network_input = preprocessing.data_preprocessing(network_input, fps)
    network_input = np.expand_dims(np.array(network_input),axis=0)

    # inference 
    human_state = inference(network_input)
    human_info.human_state = human_state

results = np.zeros((20), dtype=np.uint8)
model = load_model_for_inference(0)
