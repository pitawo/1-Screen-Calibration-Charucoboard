"""
ユーティリティ関数
"""
import os

import cv2
import numpy as np
from datetime import datetime as dt


# ガンマ補正LUT（find_chessboard_corners_robust内で再利用）
_GAMMA = 1.5
_INV_GAMMA = 1.0 / _GAMMA
_GAMMA_LUT = np.array(
    [((i / 255.0) ** _INV_GAMMA) * 255 for i in np.arange(0, 256)],
    dtype="uint8"
)


def get_video_info(video_path: str) -> dict:
    """動画の基本情報を取得"""
    cap = cv2.VideoCapture(video_path)
    info = {
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }
    cap.release()
    return info


def progress_callback(message, progress_percent=None):
    """進捗表示コールバック"""
    timestamp = dt.now().strftime("%H:%M:%S")
    if progress_percent is not None:
        print(f"[{timestamp}] {progress_percent:5.1f}% | {message}")
    else:
        print(f"[{timestamp}] {message}")


def show_chessboard_preview(video_path: str, rows: int, cols: int, output_dir: str = None) -> bool:
    """
    チェスボード検出プレビュー（画像ファイル保存方式）

    Args:
        video_path: 動画ファイルのパス
        rows: チェスボード行数（交点数）
        cols: チェスボード列数（交点数）
        output_dir: プレビュー画像の保存先

    Returns:
        bool: 検出成功したかどうか
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("エラー: 動画を開けません")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pattern_size = (cols, rows)
    chess_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

    print("\n" + "=" * 60)
    print("チェスボード検出プレビュー")
    print("=" * 60)

    # 保存先ディレクトリ
    if output_dir is None:
        output_dir = os.path.dirname(video_path) or "."
    preview_dir = os.path.join(output_dir, "preview_frames")
    os.makedirs(preview_dir, exist_ok=True)

    # サンプルフレームを抽出（最大10フレーム）
    sample_interval = max(1, total_frames // 10)
    sample_frames = list(range(0, total_frames, sample_interval))[:10]

    print(f"\n{len(sample_frames)}フレームをサンプリングして検出テスト中...")

    detected_count = 0

    for i, frame_idx in enumerate(sample_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_corners, corners = cv2.findChessboardCorners(gray, pattern_size, flags=chess_flags)

        display_frame = frame.copy()

        # ブレスコア計算
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = laplacian.var()

        if ret_corners:
            subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 1e-4)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria)
            cv2.drawChessboardCorners(display_frame, pattern_size, corners_refined, ret_corners)

            status_text = f"DETECTED | Blur: {blur_score:.1f}"
            cv2.putText(display_frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            detected_count += 1
            status = "OK"
        else:
            status_text = "NOT DETECTED"
            cv2.putText(display_frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            status = "NG"

        frame_info = f"Frame: {frame_idx + 1}/{total_frames}"
        cv2.putText(display_frame, frame_info, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # 画像保存
        filename = f"preview_{i+1:02d}_frame{frame_idx:05d}_{status}.jpg"
        filepath = os.path.join(preview_dir, filename)
        cv2.imwrite(filepath, display_frame)

        print(f"  [{i+1}/{len(sample_frames)}] Frame {frame_idx}: {status} (Blur: {blur_score:.1f})")

    cap.release()

    # 結果サマリー
    print("\n" + "-" * 40)
    print(f"検出結果: {detected_count}/{len(sample_frames)} フレームで検出成功")
    print(f"プレビュー画像保存先: {preview_dir}")
    print("-" * 40)

    if detected_count == 0:
        print("\n警告: サンプルフレームでチェスボードが検出されませんでした。")
        print("  - チェスボードのサイズ（行数・列数）が正しいか確認してください")
        print("  - 動画内にチェスボードが映っているか確認してください")

    return detected_count > 0


def get_user_input(prompt: str, default: str = None, value_type: type = str):
    """
    ユーザー入力を取得

    Args:
        prompt: プロンプト文字列
        default: デフォルト値
        value_type: 期待する型

    Returns:
        入力値（指定された型に変換）
    """
    if default is not None:
        full_prompt = f"{prompt} (デフォルト: {default}): "
    else:
        full_prompt = f"{prompt}: "

    while True:
        user_input = input(full_prompt).strip()

        if user_input == "" and default is not None:
            return value_type(default)

        if user_input == "":
            print("  → 値を入力してください")
            continue

        try:
            return value_type(user_input)
        except ValueError:
            print(f"  → 無効な入力です。{value_type.__name__}型で入力してください")


def get_yes_no(prompt: str, default: bool = True) -> bool:
    """
    Yes/No入力を取得

    Args:
        prompt: プロンプト文字列
        default: デフォルト値

    Returns:
        bool: Yesならtrue
    """
    default_str = "Y/n" if default else "y/N"
    full_prompt = f"{prompt} ({default_str}): "

    while True:
        user_input = input(full_prompt).strip().lower()

        if user_input == "":
            return default
        elif user_input in ["y", "yes"]:
            return True
        elif user_input in ["n", "no"]:
            return False
        else:
            print("  → 'y' または 'n' を入力してください")


class DualLogger:
    """
    ターミナル出力とログファイルへの書き込みを同時に行うクラス
    """
    def __init__(self):
        self.log_buffer = []

    def log(self, message: str, end: str = "\n"):
        """メッセージを出力し、バッファに保存"""
        print(message, end=end)
        self.log_buffer.append(message)

    def get_logs(self) -> list:
        """保存されたログを取得"""
        return self.log_buffer


def find_chessboard_corners_robust(gray_image, pattern_size, flags=None):
    """
    ロバストなチェスボード検出（ピンボケ・反射・照明ムラに対応）
    
    Args:
        gray_image: グレースケール画像
        pattern_size: (cols, rows)
        flags: cv2.findChessboardCorners のフラグ
        
    Returns:
        ret (bool): 検出成功か
        corners (np.ndarray): 検出されたコーナー座標
    """
    if flags is None:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

    # 1. Sector-Based Method (最もロバスト)
    # 多くのノイズ、ピンボケ、複雑な背景に強い
    sb_flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    ret, corners = cv2.findChessboardCornersSB(gray_image, pattern_size, flags=sb_flags)
    if ret:
        return True, corners

    # 2. 通常の検出（標準API）
    ret, corners = cv2.findChessboardCorners(gray_image, pattern_size, flags=flags)
    if ret:
        return True, corners

    # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # 反射や照明ムラに強い
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray_image)
    
    ret, corners = cv2.findChessboardCorners(gray_clahe, pattern_size, flags=flags)
    if ret:
        return True, corners
        
    # Sector-Based も CLAHE 画像で試行
    ret, corners = cv2.findChessboardCornersSB(gray_clahe, pattern_size, flags=sb_flags)
    if ret:
        return True, corners

    # 4. ガンマ補正（暗い/明るすぎる場合）
    gray_gamma = cv2.LUT(gray_image, _GAMMA_LUT)
    
    ret, corners = cv2.findChessboardCorners(gray_gamma, pattern_size, flags=flags)
    if ret:
        return True, corners

    return False, None
