
from dex_retargeting.retargeting_config import RetargetingConfig
from pathlib import Path
import yaml
from enum import Enum
import os

class HandType(Enum):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    INSPIRE_HAND = os.path.join(base_dir, "assets/inspire_hand/inspire_hand.yml")
    INSPIRE_HAND_Unit_Test = os.path.join(base_dir, "assets/inspire_hand/inspire_hand.yml")
    UNITREE_DEX3 = os.path.join(base_dir, "assets/unitree_hand/unitree_dex3.yml")
    UNITREE_DEX3_Unit_Test = os.path.join(base_dir, "assets/unitree_hand/unitree_dex3.yml")
    # For DexPilot, we use the same base config but load separate left/right configs
    UNITREE_DEX3_DEXPILOT_LEFT = os.path.join(base_dir, "assets/unitree_hand/unitree_dex3_left_dexpilot.yml")
    UNITREE_DEX3_DEXPILOT_RIGHT = os.path.join(base_dir, "assets/unitree_hand/unitree_dex3_right_dexpilot.yml")
    UNITREE_DEX3_DEXPILOT_LEFT_Unit_Test = os.path.join(base_dir, "assets/unitree_hand/unitree_dex3_left_dexpilot.yml")
    UNITREE_DEX3_DEXPILOT_RIGHT_Unit_Test = os.path.join(base_dir, "assets/unitree_hand/unitree_dex3_right_dexpilot.yml")

class HandRetargeting:
    def __init__(self, hand_type: HandType, retargeting_method: str = 'vector'):
        # Set up base directory and URDF path
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        RetargetingConfig.set_default_urdf_dir(os.path.join(base_dir, 'assets'))

        try:
            # Handle different configuration formats for different retargeting methods
            if retargeting_method == 'dexpilot':
                # DexPilot uses separate config files for left and right hands
                if hand_type == HandType.UNITREE_DEX3:
                    left_config_path = Path(HandType.UNITREE_DEX3_DEXPILOT_LEFT.value)
                    right_config_path = Path(HandType.UNITREE_DEX3_DEXPILOT_RIGHT.value)
                elif hand_type == HandType.UNITREE_DEX3_Unit_Test:
                    left_config_path = Path(HandType.UNITREE_DEX3_DEXPILOT_LEFT_Unit_Test .value)  # Use same configs for unit test
                    right_config_path = Path(HandType.UNITREE_DEX3_DEXPILOT_RIGHT_Unit_Test .value)
                else:
                    # For other hand types, fall back to the old path generation method
                    config_file_path = Path(hand_type.value)
                    base_name = config_file_path.stem
                    base_dir = config_file_path.parent
                    left_config_path = base_dir / f"{base_name}_left_dexpilot.yml"
                    right_config_path = base_dir / f"{base_name}_right_dexpilot.yml"
                
                print(f"DexPilot loading left config: {left_config_path}")
                print(f"DexPilot loading right config: {right_config_path}")
                
                with left_config_path.open('r') as f:
                    left_cfg = yaml.safe_load(f)
                with right_config_path.open('r') as f:
                    right_cfg = yaml.safe_load(f)
                    
                self.cfg = {'left': left_cfg, 'right': right_cfg}
                left_retargeting_config = RetargetingConfig.from_dict(left_cfg['retargeting'])
                right_retargeting_config = RetargetingConfig.from_dict(right_cfg['retargeting'])
            else:
                # Vector retargeting uses a single config file with left/right sections
                config_file_path = Path(hand_type.value)
                print(f"Vector loading config: {config_file_path}")
                with config_file_path.open('r') as f:
                    self.cfg = yaml.safe_load(f)
                    
                if 'left' not in self.cfg or 'right' not in self.cfg:
                    raise ValueError("Configuration file must contain 'left' and 'right' keys.")

                # For vector retargeting
                left_retargeting_config = RetargetingConfig.from_dict(self.cfg['left'])
                right_retargeting_config = RetargetingConfig.from_dict(self.cfg['right'])
            
            self.left_retargeting = left_retargeting_config.build()
            self.right_retargeting = right_retargeting_config.build()

            # For both DexPilot and vector retargeting, joint_names should be available
            ## ----- Macht das hier probleme?? --
            # Moved to arcive to test if this is the problem area
            # try:
            #     self.left_retargeting_joint_names = self.left_retargeting.joint_names
            #     self.right_retargeting_joint_names = self.right_retargeting.joint_names
            # except AttributeError:
            #     # Fallback for older configurations
            #     if retargeting_method == 'dexpilot':
            #         # For DexPilot, we work with fingertip positions, not joint names
            #         self.left_retargeting_joint_names = ['thumb_tip', 'index_tip', 'middle_tip']
            #         self.right_retargeting_joint_names = ['thumb_tip', 'index_tip', 'middle_tip']
            #     else:
            #         raise
            # Instead of the above, we now use the following:
            self.left_retargeting_joint_names = self.left_retargeting.joint_names
            self.right_retargeting_joint_names = self.right_retargeting.joint_names
            # ------ End of potential problem area ------
            if hand_type == HandType.UNITREE_DEX3 or hand_type == HandType.UNITREE_DEX3_Unit_Test:
                # In section "Sort by message structure" of https://support.unitree.com/home/en/G1_developer/dexterous_hand
                self.left_dex3_api_joint_names  = [ 'left_hand_thumb_0_joint', 'left_hand_thumb_1_joint', 'left_hand_thumb_2_joint',
                                                    'left_hand_middle_0_joint', 'left_hand_middle_1_joint', 
                                                    'left_hand_index_0_joint', 'left_hand_index_1_joint' ]
                self.right_dex3_api_joint_names = [ 'right_hand_thumb_0_joint', 'right_hand_thumb_1_joint', 'right_hand_thumb_2_joint',
                                                    'right_hand_middle_0_joint', 'right_hand_middle_1_joint',
                                                    'right_hand_index_0_joint', 'right_hand_index_1_joint' ]
                
                # For both DexPilot and vector retargeting, map joint names directly
                # Note: DexPilot now includes all 7 joints in target_joint_names, so it returns a full 7-element array
                self.left_dex_retargeting_to_hardware = [ self.left_retargeting_joint_names.index(name) for name in self.left_dex3_api_joint_names]
                self.right_dex_retargeting_to_hardware = [ self.right_retargeting_joint_names.index(name) for name in self.right_dex3_api_joint_names]

                # Archive: This is the joint order of the dex-retargeting library version 0.1.1.
                # print([joint.get_name() for joint in self.left_retargeting.optimizer.robot.get_active_joints()])
                # ['left_hand_thumb_0_joint', 'left_hand_thumb_1_joint', 'left_hand_thumb_2_joint', 
                #  'left_hand_middle_0_joint', 'left_hand_middle_1_joint', 
                #  'left_hand_index_0_joint', 'left_hand_index_1_joint']
                # print([joint.get_name() for joint in self.right_retargeting.optimizer.robot.get_active_joints()])
                # ['right_hand_thumb_0_joint', 'right_hand_thumb_1_joint', 'right_hand_thumb_2_joint',
                #  'right_hand_middle_0_joint', 'right_hand_middle_1_joint', 
                #  'right_hand_index_0_joint', 'right_hand_index_1_joint']
            elif hand_type == HandType.INSPIRE_HAND or hand_type == HandType.INSPIRE_HAND_Unit_Test:
                self.left_inspire_api_joint_names  = [ 'L_pinky_proximal_joint', 'L_ring_proximal_joint', 'L_middle_proximal_joint',
                                                       'L_index_proximal_joint', 'L_thumb_proximal_pitch_joint', 'L_thumb_proximal_yaw_joint' ]
                self.right_inspire_api_joint_names = [ 'R_pinky_proximal_joint', 'R_ring_proximal_joint', 'R_middle_proximal_joint',
                                                       'R_index_proximal_joint', 'R_thumb_proximal_pitch_joint', 'R_thumb_proximal_yaw_joint' ]
                self.left_dex_retargeting_to_hardware = [ self.left_retargeting_joint_names.index(name) for name in self.left_inspire_api_joint_names]
                self.right_dex_retargeting_to_hardware = [ self.right_retargeting_joint_names.index(name) for name in self.right_inspire_api_joint_names]
        
        except FileNotFoundError as e:
            config_path = getattr(e, 'filename', 'unknown config file')
            print(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            print(f"YAML error while reading configuration: {e}")
            raise
        except Exception as e:
            print(f"An error occurred while loading configuration: {e}")
            raise
