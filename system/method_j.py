"""
MethodJ_GeometricDiversity クラス
高精度自動キャリブレーション（位置・角度のバランスを考慮したフレーム選択）
"""
import math
import datetime
from typing import Optional, Callable

import cv2
import numpy as np

from .base_calibrator import BaseCalibrator
from .utils import find_chessboard_corners_robust, DualLogger


class MethodJ_GeometricDiversity(BaseCalibrator):
    """
    高精度自動キャリブレーション（位置・角度のバランスを考慮したフレーム選択）
    """

    def __init__(
        self,
        checkerboard_rows: int,
        checkerboard_cols: int,
        square_size: float,
        target_frame_count: int = 200,
        blur_threshold: float = 120.0,
        enable_k_center: bool = True,
        frame_skip: int = 1,
        logger: Optional[DualLogger] = None
    ):
        super().__init__(checkerboard_rows, checkerboard_cols, square_size)
        self.target_frame_count = max(1, int(target_frame_count))
        self.blur_threshold = blur_threshold
        self.enable_k_center = enable_k_center
        self.frame_skip = max(1, frame_skip)
        self.logger = logger
        self.min_frames_per_bin = 3
        self.max_reflection_ratio = 0.12
        self.objp_reshaped = self.objp_template.reshape(-1, 1, 3)
        self.process_log = []

    def _log(self, message: str):
        """処理ログを記録"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] {message}"
        self.process_log.append(log_entry)
        if self.logger:
             self.logger.log(log_entry)

    def _compute_blur_score(self, gray_image):
        """ブレスコアを計算（ラプラシアン分散）"""
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        variance = laplacian.var()
        return variance

    def _build_feature_matrix(self, detections):
        """k-center 用特徴量行列を構築"""
        return np.asarray([
            [
                d['center_u'],
                d['center_v'],
                np.log(d.get('scale', 0.5) + 1e-6),
                d['angle_2d'] / 90.0,
                d['slant_score']
            ]
            for d in detections
        ], dtype=np.float32)

    @staticmethod
    def _select_top_by_quality(detections, top_k):
        """品質スコア上位フレームを取得"""
        return sorted(detections, key=lambda x: -x['quality_score'])[:top_k]

    def detect_and_evaluate_frame(self, args):
        """フレーム検出と評価（Method J用: 2D特徴量ベース）"""
        frame_idx, frame, pattern_size, img_width, img_height, img_diag = args
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blur_score = self._compute_blur_score(gray)

        # ロバスト検出を使用
        ret_corners, corners = find_chessboard_corners_robust(
            gray, pattern_size, flags=self.chess_flags
        )
        if not ret_corners:
            return None

        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), self.subpix_criteria
        )

        # 基本特徴量
        center_x = np.mean(corners_refined[:, 0, 0]) / img_width
        center_y = np.mean(corners_refined[:, 0, 1]) / img_height

        x_coords = corners_refined[:, 0, 0]
        y_coords = corners_refined[:, 0, 1]
        board_width = np.max(x_coords) - np.min(x_coords)
        board_height = np.max(y_coords) - np.min(y_coords)
        board_diag = math.hypot(board_width, board_height)
        scale = board_diag / img_diag

        # 反射（白飛び）推定: チェスボード領域近傍の高輝度割合
        pad = 10
        x_min = max(0, int(np.min(x_coords)) - pad)
        x_max = min(img_width, int(np.max(x_coords)) + pad)
        y_min = max(0, int(np.min(y_coords)) - pad)
        y_max = min(img_height, int(np.max(y_coords)) + pad)
        roi = gray[y_min:y_max, x_min:x_max]
        reflection_ratio = 0.0
        if roi.size > 0:
            reflection_ratio = float(np.mean(roi >= 245))

        # コーナー配置一貫性: 行列方向の間隔CV（低いほど安定）
        corners_grid = corners_refined.reshape(self.checkerboard_rows, self.checkerboard_cols, 2)
        h_diffs = np.linalg.norm(corners_grid[:, 1:, :] - corners_grid[:, :-1, :], axis=2).ravel()
        v_diffs = np.linalg.norm(corners_grid[1:, :, :] - corners_grid[:-1, :, :], axis=2).ravel()
        spacing = np.concatenate([h_diffs, v_diffs])
        spacing_mean = float(np.mean(spacing)) if spacing.size > 0 else 0.0
        spacing_std = float(np.std(spacing)) if spacing.size > 0 else 0.0
        spacing_cv = spacing_std / (spacing_mean + 1e-6)

        # 画質評価（Laplacianはブレスコアと共通化）
        sharpness = blur_score
        contrast = gray.std()
        sharpness_norm = min(sharpness / 1000.0, 1.0)
        contrast_norm = min(contrast / 100.0, 1.0)
        geometry_norm = max(0.0, 1.0 - min(spacing_cv, 1.0))
        reflection_penalty = min(reflection_ratio / max(self.max_reflection_ratio, 1e-6), 1.0)
        quality_score = (
            sharpness_norm * 0.45 +
            contrast_norm * 0.20 +
            geometry_norm * 0.35
        ) * (1.0 - 0.4 * reflection_penalty)

        # 2D幾何特徴量の計算（Step 2, 3の代替）
        # 1. 回転角 (2D Orientation)
        rect = cv2.minAreaRect(corners_refined)
        angle_2d = rect[2]
        if rect[1][0] < rect[1][1]:
            angle_2d += 90

        # 2. スラント（傾き）の簡易指標
        # 想定アスペクト比と観測アスペクト比の乖離を利用
        # チェスボードの物理的なアスペクト比
        min_board_side = min(self.checkerboard_rows - 1, self.checkerboard_cols - 1)
        if min_board_side <= 0:
            expected_aspect = 1.0
        else:
            expected_aspect = max(self.checkerboard_rows - 1, self.checkerboard_cols - 1) / min_board_side
        
        # 観測されたバウンディングボックスのアスペクト比
        # 回転を考慮した矩形から算出
        w_rect, h_rect = rect[1]
        if w_rect == 0 or h_rect == 0:
             observed_aspect = 1.0
        else:
             observed_aspect = max(w_rect, h_rect) / min(w_rect, h_rect)
        
        # アスペクト比のズレ（=傾きの大きさ）
        slant_score = abs(observed_aspect - expected_aspect)

        return {
            'frame_idx': frame_idx,
            'objp': self.objp_reshaped,
            'imgp': corners_refined,
            'quality_score': quality_score,
            'blur_score': blur_score,
            'center_u': center_x,
            'center_v': center_y,
            'scale': scale,
            'angle_2d': angle_2d,   # 新機能: 2D回転角
            'slant_score': slant_score, # 新機能: 傾き指標
            'spacing_cv': spacing_cv,
            'reflection_ratio': reflection_ratio,
            'reprojection_error': None, # Step 3削除により未計算
            'tilt_angle': None, # 削除
            'roll_angle': None, # 削除
            'yaw_angle': None   # 削除
        }

    def _k_center_greedy(self, features, k):
        """k-center法で多様性を最大化するサンプルを選択"""
        n = len(features)
        if k >= n:
            return list(range(n))

        feature_center = np.mean(features, axis=0)
        first_center = int(np.argmax(np.linalg.norm(features - feature_center, axis=1)))
        centers = [first_center]
        distances = np.full(n, np.inf)
        selected_mask = np.zeros(n, dtype=bool)
        selected_mask[centers[0]] = True
        distances[centers[0]] = -np.inf

        for _ in range(k - 1):
            last_center = features[centers[-1]]
            remaining_indices = np.where(~selected_mask)[0]
            if len(remaining_indices) == 0:
                break

            candidate_vectors = features[remaining_indices]
            dist_to_last = np.linalg.norm(candidate_vectors - last_center, axis=1)
            distances[remaining_indices] = np.minimum(distances[remaining_indices], dist_to_last)

            next_center = int(remaining_indices[np.argmax(distances[remaining_indices])])
            centers.append(next_center)
            selected_mask[next_center] = True
            distances[next_center] = -np.inf

        return centers

    def run_calibration(
        self,
        video_path: str,
        progress_callback: Optional[Callable] = None
    ) -> dict:
        """キャリブレーション実行（メモリ効率化版: フレームを1枚ずつ処理）"""
        self.process_log = []
        self._log("=" * 60)
        self._log("キャリブレーション 開始")
        self._log("=" * 60)

        if progress_callback:
            progress_callback("動画情報を取得中...", 0)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        img_diag = math.sqrt(img_width**2 + img_height**2)
        pattern_size = (self.checkerboard_cols, self.checkerboard_rows)
        img_size = (img_width, img_height)

        process_count = len(range(0, total_frames, self.frame_skip))
        self._log(f"動画: {total_frames}フレーム, {img_width}x{img_height}")
        if self.frame_skip > 1:
            self._log(f"フレーム間引き: {self.frame_skip}フレームおき → 処理対象 {process_count}フレーム")

        # ステップ1: フレーム検出（1枚ずつ逐次処理）
        self._log("--- ステップ1: フレーム検出（ブレ判定含む） ---")
        self._log(f"ブレ判定の厳しさ: {self.blur_threshold}")
        if progress_callback:
            progress_callback("フレームを検出中...", 5)

        all_detections = []
        cap = cv2.VideoCapture(video_path)
        processed = 0
        for idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            if idx % self.frame_skip != 0:
                continue

            result = self.detect_and_evaluate_frame(
                (idx, frame, pattern_size, img_width, img_height, img_diag)
            )
            if result is not None:
                all_detections.append(result)
            processed += 1
            if progress_callback and processed % 100 == 0:
                pct = 5 + (processed / process_count) * 20
                progress_callback(f"検出中... {processed}/{process_count}", min(pct, 25))
        cap.release()

        self._log(f"検出成功: {len(all_detections)}/{process_count}フレーム ({len(all_detections)/max(process_count,1)*100:.1f}%)")

        if len(all_detections) == 0:
            raise ValueError("チェスボードが検出されたフレームがありません")

        # ブレ除外
        blur_scores = np.asarray([d['blur_score'] for d in all_detections], dtype=np.float32)
        self._log(f"ブレスコア統計: 平均={np.mean(blur_scores):.1f}, 中央値={np.median(blur_scores):.1f}, "
                 f"最小={np.min(blur_scores):.1f}, 最大={np.max(blur_scores):.1f}")

        effective_blur_threshold = self.blur_threshold
        sharp_detections = [d for d in all_detections if d['blur_score'] >= effective_blur_threshold]
        blurred_count = len(all_detections) - len(sharp_detections)
        self._log(f"ブレ除外: {len(sharp_detections)}フレーム残存（{blurred_count}フレーム除外）")

        # 候補が少なすぎる場合は、ブレしきい値を段階的に緩和して候補数を確保
        min_candidate_count = min(len(all_detections), max(40, self.target_frame_count // 2))
        if 0 < len(sharp_detections) < min_candidate_count:
            q30 = float(np.percentile(blur_scores, 30))
            relaxed_threshold = min(effective_blur_threshold, max(q30, effective_blur_threshold * 0.65))
            relaxed_detections = [d for d in all_detections if d['blur_score'] >= relaxed_threshold]
            if len(relaxed_detections) > len(sharp_detections):
                self._log(
                    f"候補不足のためブレしきい値を緩和: {effective_blur_threshold:.1f} -> {relaxed_threshold:.1f} "
                    f"({len(sharp_detections)} -> {len(relaxed_detections)}枚)"
                )
                sharp_detections = relaxed_detections
                effective_blur_threshold = relaxed_threshold

        if len(sharp_detections) == 0:
            self._log(f"警告: 指定されたブレ基準（{self.blur_threshold}）を満たすフレームがありませんが、")
            self._log("チェスボード検出には成功しているため、検出された全フレームを使用します。（頑健検出モード）")
            sharp_detections = all_detections
            effective_blur_threshold = float(np.min(blur_scores))
            
        elif len(sharp_detections) < self.min_frames_per_bin * 5: # 極端に少ない場合も救済
             self._log(f"警告: ブレ基準を満たすフレームが少なすぎます（{len(sharp_detections)}枚）。")
             self._log("検出された全フレームを候補として使用します。")
             sharp_detections = all_detections
             effective_blur_threshold = float(np.min(blur_scores))

        del all_detections  # ここ以降は sharp_detections のみ使用
        if len(sharp_detections) == 0:
             raise ValueError("チェスボードが検出できませんでした（検出数 0）")

        # 品質下位フレームを除外（ただし候補数を減らしすぎない）
        if len(sharp_detections) > self.target_frame_count:
            quality_values = np.array([d['quality_score'] for d in sharp_detections], dtype=np.float32)
            quality_floor = float(np.percentile(quality_values, 20))
            quality_filtered = [d for d in sharp_detections if d['quality_score'] >= quality_floor]
            min_after_quality = max(self.target_frame_count, int(len(sharp_detections) * 0.6))

            if len(quality_filtered) >= min_after_quality:
                self._log(
                    f"品質下位20%を除外: {len(sharp_detections)} -> {len(quality_filtered)}フレーム"
                )
                sharp_detections = quality_filtered
            else:
                self._log(
                    f"品質除外は候補不足になるためスキップ: {len(sharp_detections)}フレーム維持"
                )

        if progress_callback:
            progress_callback(f"{len(sharp_detections)}フレームがブレ判定を通過", 30)

        # ステップ2: フレーム選択（2D特徴量 + k-center法）
        self._log("--- ステップ2: フレーム選択（2D特徴量 + k-center法） ---")
        if progress_callback:
            progress_callback("フレーム選択中...", 40)

        # 全候補からk-center法で選択
        selected_frames = []
        
        target_count = self.target_frame_count
        if len(sharp_detections) <= target_count:
            selected_frames = sharp_detections
            self._log(f"全フレームを選択（候補数 {len(selected_frames)} <= 目標数 {target_count}）")
        else:
            features = self._build_feature_matrix(sharp_detections)
            
            # k-center法実行
            if self.enable_k_center and len(features) > 0:
                try:
                    selected_indices = self._k_center_greedy(features, target_count)
                    selected_frames = [sharp_detections[idx] for idx in selected_indices]
                    self._log(f"k-center法により {len(selected_frames)}フレームを選択")
                except Exception as e:
                    self._log(f"k-center法エラー: {e} -> 品質スコア順にフォールバック")
                    selected_frames = self._select_top_by_quality(sharp_detections, target_count)
            else:
                # k-center無効時は品質スコア順
                selected_frames = self._select_top_by_quality(sharp_detections, target_count)
                self._log(f"品質スコア順に {len(selected_frames)}フレームを選択")

        selected_frames.sort(key=lambda x: x['frame_idx'])
        
        # bin_id エラー回避: ダミー値をセット
        for f in selected_frames:
             f['bin_id'] = 'all'

        # ダミーのbin_countsを作成（互換性のため）
        bin_counts = {'all': selected_frames}


        # ステップ3: 最終キャリブレーション
        self._log("--- ステップ3: 最終キャリブレーション ---")
        if progress_callback:
            progress_callback("最終キャリブレーション実行中...", 50)

        total_objpoints = [d['objp'] for d in selected_frames]
        total_imgpoints = [d['imgp'] for d in selected_frames]

        result = self._finalize_calibration_j(
            total_objpoints, total_imgpoints, img_size, progress_callback,
            selected_frames, bin_counts, effective_blur_threshold
        )

        result['process_log'] = self.process_log

        return result



    def _finalize_calibration_j(self, objpoints, imgpoints, img_size, progress_callback,
                                 selected_frames, bin_counts, blur_threshold_used):
        """最終キャリブレーション処理"""
        # キャリブレーション用フレーム数を制限（大量フレームで極端に遅くなるため）
        # 単純な等間隔抽出だけだと高品質フレームを取りこぼす可能性があるため、
        # 品質優先 + 時間方向の分散を両立するサンプリングにする。
        max_calib_frames = 80
        if len(objpoints) > max_calib_frames:
            quality_ratio = 0.7
            quality_count = int(max_calib_frames * quality_ratio)
            diversity_count = max_calib_frames - quality_count

            scored = list(enumerate(selected_frames))
            top_quality = sorted(
                scored,
                key=lambda x: x[1].get('quality_score', 0.0),
                reverse=True
            )[:quality_count]
            selected_idx_set = {idx for idx, _ in top_quality}

            if diversity_count > 0:
                step = len(objpoints) / diversity_count
                uniform_indices = [min(int(i * step), len(objpoints) - 1) for i in range(diversity_count)]
                selected_idx_set.update(uniform_indices)

            indices = sorted(selected_idx_set)
            if len(indices) > max_calib_frames:
                indices = indices[:max_calib_frames]
            elif len(indices) < max_calib_frames:
                for idx in range(len(objpoints)):
                    if idx not in selected_idx_set:
                        indices.append(idx)
                        if len(indices) == max_calib_frames:
                            break

            calib_obj = [objpoints[i] for i in indices]
            calib_img = [imgpoints[i] for i in indices]
            self._log(f"最終キャリブレーション実行（候補{len(objpoints)}フレーム、上限{max_calib_frames}枚）")
            self._log(
                f"採用基準: 計算時間と安定性のバランスのため最大{max_calib_frames}枚。"
                f"超過時は品質上位{quality_count}枚 + 時系列の均等サンプル{len(calib_obj)-quality_count}枚を併用"
            )
        else:
            calib_obj = objpoints
            calib_img = imgpoints
            self._log(f"最終キャリブレーション実行（{len(objpoints)}フレーム）")

        if progress_callback:
            progress_callback(f"最終キャリブレーション実行中（{len(calib_obj)}フレーム）...", 93)

        rms, camera_matrix, dist_coeffs, rvecs, tvecs = self.calibrate_chessboard(
            calib_obj, calib_img, img_size
        )

        pv_errs = self.per_view_errors(
            calib_obj, calib_img, rvecs, tvecs, camera_matrix, dist_coeffs
        )

        # 外れ値除外で再最適化（RMS改善）
        if len(calib_obj) >= 25:
            med = float(np.median(pv_errs))
            mad = float(np.median(np.abs(pv_errs - med)))
            robust_sigma = max(1.4826 * mad, 1e-6)
            outlier_thr = med + 2.8 * robust_sigma
            keep_mask = pv_errs <= outlier_thr
            keep_count = int(np.sum(keep_mask))
            min_keep = max(20, int(len(calib_obj) * 0.75))

            if keep_count >= min_keep and keep_count < len(calib_obj):
                calib_obj_refined = [o for o, k in zip(calib_obj, keep_mask) if k]
                calib_img_refined = [i for i, k in zip(calib_img, keep_mask) if k]
                self._log(
                    f"外れ値除外: {len(calib_obj) - keep_count}枚除外（閾値={outlier_thr:.3f}）"
                )

                rms2, camera_matrix2, dist_coeffs2, rvecs2, tvecs2 = self.calibrate_chessboard(
                    calib_obj_refined, calib_img_refined, img_size
                )
                pv_errs2 = self.per_view_errors(
                    calib_obj_refined, calib_img_refined, rvecs2, tvecs2, camera_matrix2, dist_coeffs2
                )

                if rms2 <= rms:
                    self._log(f"再最適化採用: RMS {rms:.4f} -> {rms2:.4f}")
                    rms, camera_matrix, dist_coeffs, rvecs, tvecs = rms2, camera_matrix2, dist_coeffs2, rvecs2, tvecs2
                    pv_errs = pv_errs2
                else:
                    self._log(f"再最適化棄却: RMS {rms:.4f} -> {rms2:.4f}")

        final_used_frames = len(pv_errs)
        self._log(f"最終誤差: {rms:.4f}")
        self._log(f"使用フレーム数: {final_used_frames}")
        self._log(f"フレームごとの誤差: 平均={np.mean(pv_errs):.3f}, 95%タイル={np.percentile(pv_errs, 95):.3f}, 最大={np.max(pv_errs):.3f}")

        if progress_callback:
            progress_callback(f"最終誤差: {rms:.4f}", 97)

        h, w = img_size[1], img_size[0]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )

        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1
        )

        selected_count_by_bin = {}
        for frame in selected_frames:
            bin_id = frame.get('bin_id', 'unknown')
            selected_count_by_bin[bin_id] = selected_count_by_bin.get(bin_id, 0) + 1

        bin_statistics = {}
        for bin_id, frames in bin_counts.items():
            
            # 最初のフレームから属性を取得（なければデフォルト値）
            first_frame = frames[0] if frames else {}
            
            bin_statistics[bin_id] = {
                'total_frames': len(frames),
                'selected_frames': selected_count_by_bin.get(bin_id, 0),
                'is_edge': first_frame.get('is_edge', False),
                'scale_bin': first_frame.get('scale_bin', 'unknown'),
                'tilt_bin': first_frame.get('tilt_bin', 'unknown'),
                'roll_bin': first_frame.get('roll_bin', 'unknown'),
                'yaw_bin': first_frame.get('yaw_bin', 'unknown')
            }

        selected_qualities = [d['quality_score'] for d in selected_frames]
        selected_blur_scores = [d.get('blur_score', 0) for d in selected_frames]
        selected_scales = [d['scale'] for d in selected_frames]
        selected_reflections = [d.get('reflection_ratio', 0.0) for d in selected_frames]
        selected_spacing_cv = [d.get('spacing_cv', 0.0) for d in selected_frames]
        selected_tilts = [d['tilt_angle'] for d in selected_frames if d['tilt_angle'] is not None]
        selected_rolls = [d['roll_angle'] for d in selected_frames if d['roll_angle'] is not None]
        selected_yaws = [d['yaw_angle'] for d in selected_frames if d['yaw_angle'] is not None]

        frame_statistics = {
            'total_selected': len(selected_frames),
            'quality_score': {
                'mean': float(np.mean(selected_qualities)),
                'median': float(np.median(selected_qualities)),
                'std': float(np.std(selected_qualities)),
                'min': float(np.min(selected_qualities)),
                'max': float(np.max(selected_qualities))
            },
            'blur_score': {
                'mean': float(np.mean(selected_blur_scores)),
                'median': float(np.median(selected_blur_scores)),
                'std': float(np.std(selected_blur_scores)),
                'min': float(np.min(selected_blur_scores)),
                'max': float(np.max(selected_blur_scores))
            },
            'scale': {
                'mean': float(np.mean(selected_scales)),
                'median': float(np.median(selected_scales)),
                'std': float(np.std(selected_scales)),
                'min': float(np.min(selected_scales)),
                'max': float(np.max(selected_scales))
            },
            'reflection_ratio': {
                'mean': float(np.mean(selected_reflections)),
                'median': float(np.median(selected_reflections)),
                'std': float(np.std(selected_reflections)),
                'min': float(np.min(selected_reflections)),
                'max': float(np.max(selected_reflections))
            },
            'corner_spacing_cv': {
                'mean': float(np.mean(selected_spacing_cv)),
                'median': float(np.median(selected_spacing_cv)),
                'std': float(np.std(selected_spacing_cv)),
                'min': float(np.min(selected_spacing_cv)),
                'max': float(np.max(selected_spacing_cv))
            }
        }

        if selected_tilts:
            frame_statistics['tilt_angle'] = {
                'mean': float(np.mean(selected_tilts)),
                'median': float(np.median(selected_tilts)),
                'std': float(np.std(selected_tilts)),
                'min': float(np.min(selected_tilts)),
                'max': float(np.max(selected_tilts))
            }

        if selected_rolls:
            frame_statistics['roll_angle'] = {
                'mean': float(np.mean(selected_rolls)),
                'median': float(np.median(selected_rolls)),
                'std': float(np.std(selected_rolls)),
                'min': float(np.min(selected_rolls)),
                'max': float(np.max(selected_rolls))
            }

        if selected_yaws:
            frame_statistics['yaw_angle'] = {
                'mean': float(np.mean(selected_yaws)),
                'median': float(np.median(selected_yaws)),
                'std': float(np.std(selected_yaws)),
                'min': float(np.min(selected_yaws)),
                'max': float(np.max(selected_yaws))
            }

        result = {
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs,
            'rms': rms,
            'image_size': np.array([w, h]),
            'board_shape': np.array([self.checkerboard_rows, self.checkerboard_cols]),
            'square_size': self.square_size,
            'calib_flags': int(self.chess_flags),
            'per_view_errors': pv_errs,
            'calib_date': datetime.datetime.now().strftime("%Y-%m-%d"),
            'new_camera_matrix': new_camera_matrix,
            'roi': np.array(roi),
            'map1': map1,
            'map2': map2,
            'method_details': {
                'method': '高精度自動キャリブレーション（3軸対応: 2D特徴量版）',
                'lens_type': 'normal',
                'blur_threshold': blur_threshold_used,
                'enable_k_center': self.enable_k_center,
                'total_bins': len(bin_counts),
                'bin_statistics': bin_statistics,
                'frame_statistics': frame_statistics
            }
        }

        self._log("=" * 60)
        self._log(f"キャリブレーション完了: 誤差={rms:.4f}, 95%タイル={np.percentile(pv_errs, 95):.3f}")
        self._log("=" * 60)

        if progress_callback:
            progress_callback(f"完了！誤差: {rms:.4f}", 100)

        return result
