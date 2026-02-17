"""
FisheyeCalibrator / FisheyeMethodJ クラス
魚眼レンズ（GoPro等）キャリブレーション対応
cv2.fisheye API を使用
"""
import re
import math
import time
import datetime
from typing import List, Tuple, Optional, Callable

import cv2
import numpy as np

from .base_calibrator import BaseCalibrator
from .utils import find_chessboard_corners_robust, DualLogger


class FisheyeCalibrator(BaseCalibrator):
    """魚眼レンズ用キャリブレータークラス"""

    def __init__(
        self,
        checkerboard_rows: int,
        checkerboard_cols: int,
        square_size: float
    ):
        super().__init__(checkerboard_rows, checkerboard_cols, square_size)

        # 魚眼キャリブレーション用フラグ
        self.fisheye_flags = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
            cv2.fisheye.CALIB_CHECK_COND +
            cv2.fisheye.CALIB_FIX_SKEW
        )
        self.fisheye_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            1e-6
        )

    def _get_fisheye_objpoints(self, objpoints: List) -> List:
        """fisheye API 用に objpoints の shape を (N, 1, 3) に統一"""
        result = []
        for objp in objpoints:
            if objp.ndim == 2:
                result.append(objp.reshape(-1, 1, 3).astype(np.float64))
            elif objp.ndim == 3 and objp.shape[1] == 1:
                result.append(objp.astype(np.float64))
            else:
                result.append(objp.reshape(-1, 1, 3).astype(np.float64))
        return result

    def _get_fisheye_imgpoints(self, imgpoints: List) -> List:
        """fisheye API 用に imgpoints の shape を (N, 1, 2) に統一"""
        result = []
        for imgp in imgpoints:
            if imgp.ndim == 2:
                result.append(imgp.reshape(-1, 1, 2).astype(np.float64))
            elif imgp.ndim == 3 and imgp.shape[1] == 1:
                result.append(imgp.astype(np.float64))
            else:
                result.append(imgp.reshape(-1, 1, 2).astype(np.float64))
        return result

    def _is_recoverable_fisheye_error(self, error_msg: str) -> bool:
        """リトライで回復可能な魚眼キャリブレーションエラーかを判定"""
        recoverable_keywords = [
            "CALIB_CHECK_COND",
            "Ill-conditioned",
            "InitExtrinsics",
            "norm_u1",
            "CalibrateExtrinsics",
            "Assertion failed",
        ]
        return any(kw in error_msg for kw in recoverable_keywords)

    def _prefilter_degenerate_frames(self, obj_pts, img_pts, K=None):
        """ホモグラフィが退化しているフレームを事前除外（InitExtrinsics対策）

        K が与えられた場合は K_inv * H の列ノルムもチェックする
        （InitExtrinsics の fabs(norm_u1) > 0 と同等の検査）
        """
        K_inv = None
        if K is not None and K[0, 0] > 0:
            try:
                K_inv = np.linalg.inv(K)
            except np.linalg.LinAlgError:
                K_inv = None

        valid_obj = []
        valid_img = []
        removed = 0

        for obj, img in zip(obj_pts, img_pts):
            src = obj.reshape(-1, 3)[:, :2].astype(np.float64)
            dst = img.reshape(-1, 2).astype(np.float64)

            try:
                H, mask = cv2.findHomography(src, dst, method=0)
            except cv2.error:
                removed += 1
                continue

            if H is None:
                removed += 1
                continue

            # ホモグラフィの行列式チェック
            det = np.linalg.det(H)
            if abs(det) < 1e-10:
                removed += 1
                continue

            # 列ベクトルのノルムチェック
            h1 = H[:, 0]
            h2 = H[:, 1]
            if np.linalg.norm(h1) < 1e-10 or np.linalg.norm(h2) < 1e-10:
                removed += 1
                continue

            # 列ベクトルの外積チェック
            if np.linalg.norm(np.cross(h1, h2)) < 1e-10:
                removed += 1
                continue

            # K_inv を使った InitExtrinsics 相当のチェック
            if K_inv is not None:
                u1 = K_inv @ h1
                u2 = K_inv @ h2
                if np.linalg.norm(u1) < 1e-6 or np.linalg.norm(u2) < 1e-6:
                    removed += 1
                    continue
                u3 = np.cross(u1, u2)
                if np.linalg.norm(u3) < 1e-6:
                    removed += 1
                    continue

            valid_obj.append(obj)
            valid_img.append(img)

        return valid_obj, valid_img, removed

    def _estimate_initial_K(self, obj_pts, img_pts, image_size):
        """標準キャリブレーションでカメラ行列Kを事前推定（InitExtrinsics対策）"""
        # fisheye形式 (N,1,3)/(N,1,2) → 標準形式 (N,3)/(N,2) に変換
        std_obj = [o.reshape(-1, 3).astype(np.float32) for o in obj_pts]
        std_img = [i.reshape(-1, 2).astype(np.float32) for i in img_pts]

        try:
            ret, K_std, dist, _, _ = cv2.calibrateCamera(
                std_obj, std_img, image_size, None, None,
                flags=cv2.CALIB_FIX_K3
            )
            if ret > 0 and K_std[0, 0] > 0 and K_std[1, 1] > 0:
                return K_std.astype(np.float64)
        except cv2.error:
            pass

        # 標準キャリブレーションも失敗した場合はFOVベースで推定
        w, h = image_size
        focal = max(w, h)
        K_est = np.array([
            [focal, 0, w / 2.0],
            [0, focal, h / 2.0],
            [0, 0, 1.0]
        ], dtype=np.float64)
        return K_est

    def calibrate_fisheye(
        self,
        objpoints: List,
        imgpoints: List,
        image_size: Tuple[int, int]
    ) -> Tuple:
        """魚眼キャリブレーション実行（悪条件フレーム自動除外）"""
        obj_pts = self._get_fisheye_objpoints(objpoints)
        img_pts = self._get_fisheye_imgpoints(imgpoints)

        # ステージ1: 基本的な事前フィルタ（Kなし）
        obj_pts, img_pts, prefilter_removed = self._prefilter_degenerate_frames(obj_pts, img_pts)
        if prefilter_removed > 0 and hasattr(self, '_log'):
            self._log(f"  事前フィルタ: 退化フレーム {prefilter_removed}枚を除外（残り{len(obj_pts)}枚）")

        if len(obj_pts) < 10:
            raise ValueError(
                f"キャリブレーション失敗: 事前フィルタ後の有効フレームが不足しています（{len(obj_pts)}枚）"
            )

        # ステージ2: 標準キャリブレーションでK推定 → K考慮の事前フィルタ
        K_est = self._estimate_initial_K(obj_pts, img_pts, image_size)
        if hasattr(self, '_log'):
            self._log(f"  K事前推定: fx={K_est[0,0]:.1f}, fy={K_est[1,1]:.1f}, cx={K_est[0,2]:.1f}, cy={K_est[1,2]:.1f}")

        obj_pts, img_pts, k_removed = self._prefilter_degenerate_frames(obj_pts, img_pts, K=K_est)
        if k_removed > 0 and hasattr(self, '_log'):
            self._log(f"  K考慮フィルタ: 退化フレーム {k_removed}枚を追加除外（残り{len(obj_pts)}枚）")

        if len(obj_pts) < 10:
            raise ValueError(
                f"キャリブレーション失敗: フィルタ後の有効フレームが不足しています（{len(obj_pts)}枚）"
            )

        total_prefilter = prefilter_removed + k_removed

        # ステージ3: K初期値付きで魚眼キャリブレーション試行
        removed_count = 0
        max_removals = min(max(1, len(obj_pts) // 2), 20)  # 除外上限を20枚に制限
        stage3_start = time.time()
        stage3_timeout = 120  # 2分でタイムアウト

        while len(obj_pts) >= 10:
            # タイムアウト判定
            if time.time() - stage3_start > stage3_timeout:
                if hasattr(self, '_log'):
                    self._log(f"  ステージ3タイムアウト（{stage3_timeout}秒超過、{removed_count}枚除外済み）、フォールバックへ")
                break

            K = K_est.copy()
            D = np.zeros((4, 1))
            flags = self.fisheye_flags | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
            try:
                retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                    objectPoints=obj_pts,
                    imagePoints=img_pts,
                    image_size=image_size,
                    K=K,
                    D=D,
                    flags=flags,
                    criteria=self.fisheye_criteria
                )
                total_removed = total_prefilter + removed_count
                if total_removed > 0 and hasattr(self, '_log'):
                    self._log(f"  悪条件フレーム 計{total_removed}枚を除外してキャリブレーション成功")
                return retval, K, D, rvecs, tvecs
            except cv2.error as e:
                error_msg = str(e)
                if not self._is_recoverable_fisheye_error(error_msg):
                    raise
                if removed_count >= max_removals:
                    if hasattr(self, '_log'):
                        self._log(f"  除外上限（{max_removals}枚）到達、フォールバックへ")
                    break

                # エラーメッセージからフレーム番号を取得できる場合
                match = re.search(r'input array (\d+)', error_msg)
                if match:
                    bad_idx = int(match.group(1))
                    if bad_idx < len(obj_pts):
                        obj_pts.pop(bad_idx)
                        img_pts.pop(bad_idx)
                        removed_count += 1
                        if hasattr(self, '_log'):
                            self._log(f"  悪条件フレーム検出（index={bad_idx}）、除外して再試行")
                        continue

                # フレーム番号不明 → フォールバックへ
                if hasattr(self, '_log'):
                    self._log(f"  フレーム特定不可のエラー: {error_msg[:80]}")
                break

        # ステージ4: フォールバック（CALIB_CHECK_COND除去 + K初期値）
        if hasattr(self, '_log'):
            self._log("  フォールバック: CALIB_CHECK_COND除去 + K初期値で実行")

        flags_fallback = (
            (self.fisheye_flags & ~cv2.fisheye.CALIB_CHECK_COND)
            | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
        )

        K = K_est.copy()
        D = np.zeros((4, 1))

        try:
            retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                objectPoints=obj_pts,
                imagePoints=img_pts,
                image_size=image_size,
                K=K,
                D=D,
                flags=flags_fallback,
                criteria=self.fisheye_criteria
            )
        except cv2.error:
            # RECOMPUTE_EXTRINSICも除去して最小フラグで試行
            if hasattr(self, '_log'):
                self._log("  最小フラグで再試行（RECOMPUTE_EXTRINSIC + CHECK_COND 除去）")
            flags_minimal = cv2.fisheye.CALIB_FIX_SKEW | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
            K = K_est.copy()
            D = np.zeros((4, 1))
            try:
                retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                    objectPoints=obj_pts,
                    imagePoints=img_pts,
                    image_size=image_size,
                    K=K,
                    D=D,
                    flags=flags_minimal,
                    criteria=self.fisheye_criteria
                )
            except cv2.error as e3:
                raise ValueError(f"キャリブレーション失敗（全フォールバック失敗）: {e3}")

        # ステージ5: per-view誤差で外れ値除外 → 再キャリブレーション
        if hasattr(self, '_log'):
            self._log(f"  フォールバック成功（誤差={retval:.4f}）、外れ値除外して再実行")

        pv_errs = []
        for obj, img, r, t in zip(obj_pts, img_pts, rvecs, tvecs):
            proj, _ = cv2.fisheye.projectPoints(
                objectPoints=obj, rvec=r, tvec=t, K=K, D=D
            )
            e = cv2.norm(img, proj, cv2.NORM_L2) / np.sqrt(len(obj))
            pv_errs.append(e)
        pv_errs = np.array(pv_errs)

        threshold = np.percentile(pv_errs, 95)
        good_obj = []
        good_img = []
        outlier_count = 0
        for i, (obj, img) in enumerate(zip(obj_pts, img_pts)):
            if pv_errs[i] <= threshold:
                good_obj.append(obj)
                good_img.append(img)
            else:
                outlier_count += 1

        if hasattr(self, '_log'):
            self._log(f"  外れ値フレーム {outlier_count}枚を除外（残り{len(good_obj)}枚）")

        if len(good_obj) < 10:
            if hasattr(self, '_log'):
                self._log("  外れ値除外後のフレーム不足、フォールバック結果をそのまま使用")
            return retval, K, D, rvecs, tvecs

        # 外れ値除外後に退化フレームを再フィルタリング
        good_obj, good_img, refilter_removed = self._prefilter_degenerate_frames(good_obj, good_img, K=K)
        if refilter_removed > 0 and hasattr(self, '_log'):
            self._log(f"  外れ値除外後の退化フレーム再フィルタ: {refilter_removed}枚除外（残り{len(good_obj)}枚）")

        if len(good_obj) < 10:
            if hasattr(self, '_log'):
                self._log("  再フィルタ後のフレーム不足、フォールバック結果をそのまま使用")
            return retval, K, D, rvecs, tvecs

        # 外れ値除外後にK初期値付きで再キャリブレーション
        K2 = K.copy()
        D2 = np.zeros((4, 1))
        flags_retry = self.fisheye_flags | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
        try:
            retval2, K2, D2, rvecs2, tvecs2 = cv2.fisheye.calibrate(
                objectPoints=good_obj,
                imagePoints=good_img,
                image_size=image_size,
                K=K2,
                D=D2,
                flags=flags_retry,
                criteria=self.fisheye_criteria
            )
            if hasattr(self, '_log'):
                self._log(f"  外れ値除外後の再キャリブレーション成功（誤差={retval2:.4f}）")
            return retval2, K2, D2, rvecs2, tvecs2
        except cv2.error:
            try:
                K2 = K.copy()
                D2 = np.zeros((4, 1))
                retval2, K2, D2, rvecs2, tvecs2 = cv2.fisheye.calibrate(
                    objectPoints=good_obj,
                    imagePoints=good_img,
                    image_size=image_size,
                    K=K2,
                    D=D2,
                    flags=flags_fallback,
                    criteria=self.fisheye_criteria
                )
                if hasattr(self, '_log'):
                    self._log(f"  CALIB_CHECK_CONDなしで再キャリブレーション成功（誤差={retval2:.4f}）")
                return retval2, K2, D2, rvecs2, tvecs2
            except cv2.error:
                try:
                    K2 = K.copy()
                    D2 = np.zeros((4, 1))
                    flags_minimal = cv2.fisheye.CALIB_FIX_SKEW | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
                    retval2, K2, D2, rvecs2, tvecs2 = cv2.fisheye.calibrate(
                        objectPoints=good_obj,
                        imagePoints=good_img,
                        image_size=image_size,
                        K=K2,
                        D=D2,
                        flags=flags_minimal,
                        criteria=self.fisheye_criteria
                    )
                    if hasattr(self, '_log'):
                        self._log(f"  最小フラグで再キャリブレーション成功（誤差={retval2:.4f}）")
                    return retval2, K2, D2, rvecs2, tvecs2
                except cv2.error:
                    if hasattr(self, '_log'):
                        self._log("  再キャリブレーション全失敗、フォールバック結果をそのまま使用")
                    return retval, K, D, rvecs, tvecs

    def per_view_errors_fisheye(
        self,
        objpoints: List,
        imgpoints: List,
        rvecs: List,
        tvecs: List,
        K: np.ndarray,
        D: np.ndarray
    ) -> np.ndarray:
        """魚眼モデルの画像ごとのRMS誤差を計算"""
        obj_pts = self._get_fisheye_objpoints(objpoints)
        img_pts = self._get_fisheye_imgpoints(imgpoints)

        errs = []
        for obj, img, r, t in zip(obj_pts, img_pts, rvecs, tvecs):
            proj, _ = cv2.fisheye.projectPoints(
                objectPoints=obj,
                rvec=r,
                tvec=t,
                K=K,
                D=D
            )
            e = cv2.norm(img, proj, cv2.NORM_L2) / np.sqrt(len(obj))
            errs.append(float(e))
        return np.array(errs, dtype=np.float32)


class FisheyeMethodJ(FisheyeCalibrator):
    """
    魚眼レンズ用 高精度自動キャリブレーション（位置・角度のバランスを考慮したフレーム選択）

    grid_size:
        1: 分割なし（画面全体を1グリッド）
        2: 2×2分割
        3: 3×3分割
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
        self.target_frame_count = target_frame_count
        self.blur_threshold = blur_threshold
        self.enable_k_center = enable_k_center
        self.frame_skip = max(1, frame_skip)
        self.logger = logger

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
        return laplacian.var()

    def detect_and_evaluate_frame(self, args):
        """フレーム検出と評価（魚眼版: 2D特徴量ベース）"""
        frame_idx, frame, pattern_size, img_width, img_height, img_diag = args
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blur_score = self._compute_blur_score(gray)

        ret_corners, corners = find_chessboard_corners_robust(
            gray, pattern_size, flags=self.chess_flags
        )
        if not ret_corners:
            return None

        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), self.subpix_criteria
        )

        center_x = np.mean(corners_refined[:, 0, 0]) / img_width
        center_y = np.mean(corners_refined[:, 0, 1]) / img_height

        x_coords = corners_refined[:, 0, 0]
        y_coords = corners_refined[:, 0, 1]
        board_width = np.max(x_coords) - np.min(x_coords)
        board_height = np.max(y_coords) - np.min(y_coords)
        board_diag = math.sqrt(board_width**2 + board_height**2)
        scale = board_diag / img_diag

        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast = gray.std()
        sharpness_norm = min(sharpness / 1000.0, 1.0)
        contrast_norm = min(contrast / 100.0, 1.0)
        quality_score = sharpness_norm * 0.6 + contrast_norm * 0.4

        # 2D幾何特徴量の計算（Step 2, 3の代替）
        # 1. 回転角 (2D Orientation)
        rect = cv2.minAreaRect(corners_refined)
        angle_2d = rect[2]
        if rect[1][0] < rect[1][1]:
            angle_2d += 90

        # 2. スラント（傾き）の簡易指標
        # 想定アスペクト比と観測アスペクト比の乖離を利用
        expected_aspect = max(self.checkerboard_rows - 1, self.checkerboard_cols - 1) / \
                          min(self.checkerboard_rows - 1, self.checkerboard_cols - 1)
        
        # 観測されたバウンディングボックスのアスペクト比
        w_rect, h_rect = rect[1]
        if w_rect == 0 or h_rect == 0:
             observed_aspect = 1.0
        else:
             observed_aspect = max(w_rect, h_rect) / min(w_rect, h_rect)
        
        slant_score = abs(observed_aspect - expected_aspect)

        return {
            'frame_idx': frame_idx,
            'objp': self.objp_template.reshape(-1, 1, 3),
            'imgp': corners_refined,
            'quality_score': quality_score,
            'blur_score': blur_score,
            'center_u': center_x,
            'center_v': center_y,
            'scale': scale,
            'angle_2d': angle_2d,
            'slant_score': slant_score,
            'reprojection_error': None,
            'rvec': None,
            'tvec': None,
            'tilt_angle': None,
            'roll_angle': None,
            'yaw_angle': None
        }

    def _k_center_greedy(self, features, k):
        """k-center法で多様性を最大化するサンプルを選択"""
        n = len(features)
        if k >= n:
            return list(range(n))

        centers = [np.random.randint(n)]
        distances = np.full(n, np.inf)

        for _ in range(k - 1):
            last_center = features[centers[-1]]
            for i in range(n):
                if i not in centers:
                    dist = np.linalg.norm(features[i] - last_center)
                    distances[i] = min(distances[i], dist)

            next_center = np.argmax(distances)
            centers.append(next_center)

        return centers

    def run_calibration(
        self,
        video_path: str,
        progress_callback: Optional[Callable] = None
    ) -> dict:
        """魚眼キャリブレーション実行（メモリ効率化版: フレームを1枚ずつ処理）"""
        self.process_log = []
        t_total_start = time.time()
        self._log("=" * 60)
        self._log("魚眼レンズ キャリブレーション 開始")
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
        t_step1_start = time.time()
        self._log("--- ステップ1: フレーム検出（ブレ判定含む） ---")
        self._log(f"ブレ判定の厳しさ: {self.blur_threshold}")
        if progress_callback:
            progress_callback("フレームを検出中...", 5)

        all_detections = []
        cap = cv2.VideoCapture(video_path)
        processed = 0
        for idx in range(0, total_frames, self.frame_skip):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
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

        t_step1_elapsed = time.time() - t_step1_start
        self._log(f"検出成功: {len(all_detections)}/{process_count}フレーム ({len(all_detections)/max(process_count,1)*100:.1f}%)")
        self._log(f"ステップ1 所要時間: {t_step1_elapsed:.1f}秒")

        if len(all_detections) == 0:
            raise ValueError("チェスボードが検出されたフレームがありません")

        # ブレ除外
        blur_scores = [d['blur_score'] for d in all_detections]
        self._log(f"ブレスコア統計: 平均={np.mean(blur_scores):.1f}, 中央値={np.median(blur_scores):.1f}, "
                 f"最小={np.min(blur_scores):.1f}, 最大={np.max(blur_scores):.1f}")

        sharp_detections = [d for d in all_detections if d['blur_score'] >= self.blur_threshold]
        blurred_count = len(all_detections) - len(sharp_detections)
        self._log(f"ブレ除外: {len(sharp_detections)}フレーム残存（{blurred_count}フレーム除外）")

        if len(sharp_detections) == 0:
            self._log(f"警告: 指定されたブレ基準（{self.blur_threshold}）を満たすフレームがありませんが、")
            self._log("チェスボード検出には成功しているため、検出された全フレームを使用します。（頑健検出モード）")
            sharp_detections = all_detections
            
        elif len(sharp_detections) < self.min_frames_per_bin * 5: # 極端に少ない場合も救済
             self._log(f"警告: ブレ基準を満たすフレームが少なすぎます（{len(sharp_detections)}枚）。")
             self._log("検出された全フレームを候補として使用します。")
             sharp_detections = all_detections

        del all_detections  # メモリ解放

        if len(sharp_detections) == 0:
            raise ValueError("チェスボードが検出できませんでした（検出数 0）")

        if progress_callback:
            progress_callback(f"{len(sharp_detections)}フレームがブレ判定を通過", 30)

        # ステップ2: フレーム選択（2D特徴量 + k-center法）
        t_step2_start = time.time()
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
            # 特徴量抽出（多様性確保のため）
            # [center_u, center_v, scale, angle, slant]
            features = []
            
            for d in sharp_detections:
                # Noneチェック
                scale = d.get('scale', 0.5)
                
                # 魚眼はスケールの重要性が高いので重み付け調整
                feat = [
                    d['center_u'], 
                    d['center_v'],
                    np.log(scale + 1e-6) * 1.5,
                    d['angle_2d'] / 90.0,
                    d['slant_score']
                ]
                features.append(feat)
                
            features = np.array(features)
            
            if self.enable_k_center and len(features) > 0:
                try:
                    selected_indices = self._k_center_greedy(features, target_count)
                    selected_frames = [sharp_detections[idx] for idx in selected_indices]
                    self._log(f"k-center法により {len(selected_frames)}フレームを選択")
                except Exception as e:
                     self._log(f"k-center法エラー: {e} -> 品質スコア順にフォールバック")
                     sorted_by_quality = sorted(sharp_detections, key=lambda x: -x['quality_score'])
                     selected_frames = sorted_by_quality[:target_count]
            else:
                 # 品質スコア順
                sorted_by_quality = sorted(sharp_detections, key=lambda x: -x['quality_score'])
                selected_frames = sorted_by_quality[:target_count]
                self._log(f"品質スコア順に {len(selected_frames)}フレームを選択")

        selected_frames.sort(key=lambda x: x['frame_idx'])

            
        # bin_id エラー回避: ダミー値をセット
        for f in selected_frames:
             f['bin_id'] = 'all'
             
        # ダミーのbin_countsを作成（互換性のため）
        bin_counts = {'all': selected_frames}

        t_step2_elapsed = time.time() - t_step2_start
        self._log(f"ステップ2 所要時間: {t_step2_elapsed:.1f}秒")

        if progress_callback:
            progress_callback("フレーム選択完了", 90)

        # ステップ3: 最終キャリブレーション（魚眼）
        t_step3_start = time.time()
        self._log("--- ステップ3: 最終キャリブレーション（魚眼モデル） ---")
        if progress_callback:
            progress_callback("最終キャリブレーション実行中（魚眼）...", 92)

        total_objpoints = [d['objp'] for d in selected_frames]
        total_imgpoints = [d['imgp'] for d in selected_frames]

        result = self._finalize_calibration_fisheye(
            total_objpoints, total_imgpoints, img_size, progress_callback,
            selected_frames, bin_counts
        )

        t_step3_elapsed = time.time() - t_step3_start
        self._log(f"ステップ3 所要時間: {t_step3_elapsed:.1f}秒")

        t_total_elapsed = time.time() - t_total_start
        self._log("=" * 60)
        self._log("解析時間の内訳:")
        self._log(f"  ステップ1 フレーム検出:       {t_step1_elapsed:>8.1f}秒")
        self._log(f"  ステップ2 フレーム選択:        {t_step2_elapsed:>8.1f}秒")
        self._log(f"  ステップ3 最終キャリブレーション: {t_step3_elapsed:>8.1f}秒")
        self._log(f"  合計:                        {t_total_elapsed:>8.1f}秒 ({t_total_elapsed/60:.1f}分)")
        self._log("=" * 60)

        result['process_log'] = self.process_log
        result['elapsed_time'] = {
            'step1_detection': round(t_step1_elapsed, 1),
            'step2_frame_selection': round(t_step2_elapsed, 1),
            'step3_final_calibration': round(t_step3_elapsed, 1),
            'total': round(t_total_elapsed, 1)
        }

        return result




    def _finalize_calibration_fisheye(self, objpoints, imgpoints, img_size, progress_callback,
                                       selected_frames, bin_counts):
        """最終キャリブレーション処理（魚眼）"""
        # キャリブレーション用フレーム数を制限（cv2.fisheye.calibrateは大量フレームで極端に遅い）
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
            self._log(f"最終キャリブレーション実行（魚眼モデル、候補{len(objpoints)}フレーム、上限{max_calib_frames}枚）")
            self._log(
                f"採用基準: 計算時間と安定性のバランスのため最大{max_calib_frames}枚。"
                f"超過時は品質上位{quality_count}枚 + 時系列の均等サンプル{len(calib_obj)-quality_count}枚を併用"
            )
        else:
            calib_obj = objpoints
            calib_img = imgpoints
            self._log(f"最終キャリブレーション実行（魚眼モデル、{len(objpoints)}フレーム）")

        if progress_callback:
            progress_callback(f"最終キャリブレーション実行中（{len(calib_obj)}フレーム）...", 93)

        rms, K, D, rvecs, tvecs = self.calibrate_fisheye(
            calib_obj, calib_img, img_size
        )

        pv_errs = self.per_view_errors_fisheye(
            calib_obj, calib_img, rvecs, tvecs, K, D
        )

        self._log(f"最終誤差（魚眼）: {rms:.4f}")
        self._log(f"使用フレーム数: {len(objpoints)}")
        self._log(f"フレームごとの誤差: 平均={np.mean(pv_errs):.3f}, 95%タイル={np.percentile(pv_errs, 95):.3f}, 最大={np.max(pv_errs):.3f}")

        if progress_callback:
            progress_callback(f"最終誤差（魚眼）: {rms:.4f}", 97)

        h, w = img_size[1], img_size[0]

        # 魚眼用の歪み補正マップ生成
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (w, h), np.eye(3), balance=1.0
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), new_K, (w, h), cv2.CV_32FC1
        )

        # bin統計
        bin_statistics = {}
        for bin_id, frames in bin_counts.items():
            selected_in_bin = [f for f in selected_frames if f.get('bin_id') == bin_id]
            
            # 最初のフレームから属性を取得（なければデフォルト値）
            first_frame = frames[0] if frames else {}
            
            bin_statistics[bin_id] = {
                'total_frames': len(frames),
                'selected_frames': len(selected_in_bin),
                'is_edge': first_frame.get('is_edge', False),
                'scale_bin': first_frame.get('scale_bin', 'unknown'),
                'tilt_bin': first_frame.get('tilt_bin', 'unknown'),
                'roll_bin': first_frame.get('roll_bin', 'unknown'),
                'yaw_bin': first_frame.get('yaw_bin', 'unknown')
            }

        selected_qualities = [d['quality_score'] for d in selected_frames]
        selected_blur_scores = [d.get('blur_score', 0) for d in selected_frames]
        selected_scales = [d['scale'] for d in selected_frames]
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
            'camera_matrix': K,
            'dist_coeffs': D,
            'rms': rms,
            'image_size': np.array([w, h]),
            'board_shape': np.array([self.checkerboard_rows, self.checkerboard_cols]),
            'square_size': self.square_size,
            'calib_flags': int(self.fisheye_flags),
            'per_view_errors': pv_errs,
            'calib_date': datetime.datetime.now().strftime("%Y-%m-%d"),
            'new_camera_matrix': new_K,
            'map1': map1,
            'map2': map2,
            'method_details': {
                'method': '魚眼レンズ 高精度自動キャリブレーション（3軸対応: 2D特徴量版）',
                'lens_type': 'fisheye',
                'blur_threshold': self.blur_threshold,
                'enable_k_center': self.enable_k_center,
                'total_bins': len(bin_counts),
                'bin_statistics': bin_statistics,
                'frame_statistics': frame_statistics
            }
        }

        self._log("=" * 60)
        self._log(f"魚眼キャリブレーション完了: 誤差={rms:.4f}, 95%タイル={np.percentile(pv_errs, 95):.3f}")
        self._log("=" * 60)

        if progress_callback:
            progress_callback(f"完了！誤差（魚眼）: {rms:.4f}", 100)

        return result
