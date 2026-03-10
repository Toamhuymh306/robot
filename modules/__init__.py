"""
TicTacToe Robot Modules Package
Smart robotic arm playing tic-tac-toe based on unsupervised learning using K-means clustering
"""

from .camera_input import CameraInput
from .image_preprocessing import ImagePreprocessor
from .board_detection import BoardDetector
from .feature_extraction import FeatureExtractor
from .kmeans_classifier import KMeansClassifier
from .board_state import BoardState
from .game_ai import GameAI
from .coordinate_mapping import CoordinateMapper
from .robot_control import RobotController

__all__ = [
    'CameraInput',
    'ImagePreprocessor',
    'BoardDetector',
    'FeatureExtractor',
    'KMeansClassifier',
    'BoardState',
    'GameAI',
    'CoordinateMapper',
    'RobotController'
]

__version__ = '1.0.0'
__author__ = 'AI & Robotics Research'
