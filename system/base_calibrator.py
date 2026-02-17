"""
BaseCalibrator クラス
基底キャリブレータークラス（通常レンズ用）
"""
import json
from typing import List, Tuple, Optional

import cv2
import numpy as np
from datetime import datetime as dt


class BaseCalibrator:
    """基底キャリブレータークラス"""

    def __init__(
        self,
        checkerboard_rows: int,
        checkerboard_cols: int,
        square_size: float
    ):
        self.checkerboard_rows = checkerboard_rows
        self.checkerboard_cols = checkerboard_cols
        self.square_size = square_size

        self.objp_template = np.zeros(
            (checkerboard_rows * checkerboard_cols, 3),
            np.float32
        )
        self.objp_template[:, :2] = np.mgrid[
            0:checkerboard_cols,
            0:checkerboard_rows
        ].T.reshape(-1, 2) * square_size

        self.chess_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        self.subpix_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            80,
            1e-4
        )

    def preload_video_frames(self, video_path: str) -> List[np.ndarray]:
        """動画フレームを全読み込み"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def detect_chessboard_from_frame(self, args: Tuple) -> Optional[Tuple]:
        """1フレームからチェスボードを検出"""
        frame_idx, frame, pattern_size = args
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret_corners, corners = cv2.findChessboardCorners(
            gray, pattern_size, flags=self.chess_flags
        )

        if ret_corners:
            corners_refined = cv2.cornerSubPix(
                gray, corners,
                winSize=(11, 11),
                zeroZone=(-1, -1),
                criteria=self.subpix_criteria
            )
            return (
                frame_idx,
                self.objp_template.reshape(-1, 1, 3),
                corners_refined,
                gray.shape[::-1]
            )
        return None

    def calibrate_chessboard(
        self,
        objpoints: List,
        imgpoints: List,
        image_size: Tuple[int, int]
    ) -> Tuple:
        """キャリブレーション実行"""
        retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objectPoints=objpoints,
            imagePoints=imgpoints,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None
        )
        return retval, camera_matrix, dist_coeffs, rvecs, tvecs

    def per_view_errors(
        self,
        objpoints: List,
        imgpoints: List,
        rvecs: List,
        tvecs: List,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray
    ) -> np.ndarray:
        """画像ごとのRMS誤差を計算"""
        errs = []
        for obj, img, r, t in zip(objpoints, imgpoints, rvecs, tvecs):
            proj, _ = cv2.projectPoints(obj, r, t, camera_matrix, dist_coeffs)
            e = cv2.norm(img, proj, cv2.NORM_L2) / np.sqrt(len(obj))
            errs.append(float(e))
        return np.array(errs, dtype=np.float32)

    def save_calibration(self, result: dict, output_path: str):
        """キャリブレーション結果を保存"""
        save_data = result.copy()

        if 'cluster_sample_frames' in save_data:
            del save_data['cluster_sample_frames']

        if 'process_log' in save_data:
            del save_data['process_log']

        if 'clustering_details' in save_data and save_data['clustering_details'] is not None:
            clustering_json = json.dumps(save_data['clustering_details'])
            save_data['clustering_details_json'] = clustering_json
            del save_data['clustering_details']

        if 'method_details' in save_data and save_data['method_details'] is not None:
            method_details_json = json.dumps(save_data['method_details'])
            save_data['method_details_json'] = method_details_json
            del save_data['method_details']

        np.savez(output_path, **save_data)

        # テキストファイルとしても保存（人間が読める形式）
        txt_path = output_path.replace('.npz', '_result.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=== Camera Calibration Result ===\n\n")
            f.write(f"Date: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            if 'rms' in result:
                f.write(f"RMS Error: {result['rms']:.6f}\n")
            
            f.write("\n--- Camera Matrix (K) ---\n")
            if 'camera_matrix' in result:
                mat = result['camera_matrix']
                if isinstance(mat, np.ndarray):
                    for row in mat:
                        f.write(f"{row}\n")
                else:
                    f.write(str(mat) + "\n")
            
            f.write("\n--- Distortion Coefficients (D) ---\n")
            if 'dist_coeffs' in result:
                dist = result['dist_coeffs']
                if isinstance(dist, np.ndarray):
                    f.write(str(dist.flatten()) + "\n")
                else:
                    f.write(str(dist) + "\n")
            
            if 'rvecs' in result:
                f.write(f"\nExtrinsics: {len(result['rvecs'])} views\n")
            
            f.write("\n=================================\n")
