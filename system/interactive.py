"""
対話型インターフェース
レンズタイプ選択・画面分割設定を含む
"""
import os
import time
from datetime import datetime as dt

import numpy as np

from .method_j import MethodJ_GeometricDiversity
from .fisheye_calibrator import FisheyeMethodJ

from .utils import (
    get_video_info,
    progress_callback,
    show_chessboard_preview,
    get_user_input,
    get_yes_no,
    DualLogger,
)







def interactive_mode_no_grid():
    """対話型インターフェース（分割なし専用）でキャリブレーションを実行"""
    logger = DualLogger()

    logger.log("\n" + "=" * 60)
    logger.log("  カメラキャリブレーション（分割なし）")
    logger.log("  対話型インターフェース")
    logger.log("=" * 60)

    # === Step 1: 動画ファイルのパス入力 ===
    logger.log("\n【Step 1】動画ファイルの指定")
    logger.log("-" * 40)

    while True:
        video_path = input("動画ファイルのパスを入力してください: ").strip()
        logger.log(f"入力された動画ファイル: {video_path}")  # 入力値をログに記録

        # 引用符を除去
        video_path = video_path.strip('"').strip("'")

        if not os.path.exists(video_path):
            logger.log(f"  → ファイルが見つかりません: {video_path}")
            continue

        if not video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
            logger.log("  → サポートされていないファイル形式です")
            continue

        break

    # 動画情報表示
    video_info = get_video_info(video_path)
    logger.log("\n動画情報:")
    logger.log(f"  ファイル: {os.path.basename(video_path)}")
    logger.log(f"  解像度: {video_info['width']}x{video_info['height']}")
    logger.log(f"  FPS: {video_info['fps']:.2f}")
    logger.log(f"  フレーム数: {video_info['frame_count']}")

    # === Step 2: レンズタイプ選択 ===
    logger.log("\n【Step 2】レンズタイプの選択")
    logger.log("-" * 40)
    logger.log("  1: 通常レンズ")
    logger.log("  2: 魚眼レンズ（GoPro等）")

    while True:
        lens_choice = get_user_input("レンズタイプを選択してください", default="1", value_type=int)
        logger.log(f"レンズタイプを選択してください: {lens_choice}")
        if lens_choice in [1, 2]:
            break
        logger.log("  → 1 または 2 を入力してください")

    is_fisheye = (lens_choice == 2)
    lens_label = "魚眼レンズ" if is_fisheye else "通常レンズ"
    logger.log(f"\n選択: {lens_label}")

    # === Step 3: チェスボードパラメータ ===
    logger.log("\n【Step 3】チェスボードパラメータ")
    logger.log("-" * 40)
    logger.log("※ 交点の数を入力してください（マスの数 - 1）")

    rows = get_user_input("行数（交点数）", default="5", value_type=int)
    logger.log(f"行数（交点数）: {rows}")
    
    cols = get_user_input("列数（交点数）", default="8", value_type=int)
    logger.log(f"列数（交点数）: {cols}")
    
    square_size = get_user_input("マスのサイズ [m]", default="0.025", value_type=float)
    logger.log(f"マスのサイズ [m]: {square_size}")

    logger.log(f"\n設定: {rows}行 x {cols}列, マスサイズ {square_size}m")

    # === Step 4: チェスボード検出プレビュー ===
    logger.log("\n【Step 4】チェスボード検出プレビュー")
    logger.log("-" * 40)

    if get_yes_no("チェスボード検出プレビューを表示しますか？", default=True):
        logger.log("チェスボード検出プレビューを表示しますか？: Yes")
        # プレビュー画像の保存先（動画と同じディレクトリ）
        preview_output_dir = os.path.dirname(video_path) or "."
        preview_ok = show_chessboard_preview(video_path, rows, cols, output_dir=preview_output_dir)

        if not preview_ok:
            if not get_yes_no("\nチェスボードが検出されませんでした。設定を変更しますか？", default=True):
                logger.log("チェスボードが検出されませんでした。設定を変更しますか？: No")
                logger.log("\n続行します。")
            else:
                logger.log("チェスボードが検出されませんでした。設定を変更しますか？: Yes")
                logger.log("\nキャンセルしました。行数・列数を確認してください。")
                return None
    else:
        logger.log("チェスボード検出プレビューを表示しますか？: No")

    # === Step 5: キャリブレーションパラメータ ===
    logger.log("\n【Step 5】キャリブレーション設定")
    logger.log("-" * 40)
    logger.log("※ Enterキーでデフォルト値を使用します\n")

    # 基本パラメータ
    logger.log("--- 基本設定 ---")
    target_frame_count = get_user_input("目標フレーム数 (150-300推奨)", default="200", value_type=int)
    logger.log(f"目標フレーム数: {target_frame_count}")
    
    blur_threshold = get_user_input("ブレ判定の厳しさ (高いほど厳しい、80-200推奨)", default="120", value_type=float)
    logger.log(f"ブレ判定の厳しさ: {blur_threshold}")

    # フレーム自動補完（有効）
    enable_k_center = True

    # フレーム間引き設定
    logger.log("\n--- フレーム間引き設定 ---")
    logger.log("メモリ不足の場合、フレームを間引いて処理できます")
    logger.log("  1: 全フレーム処理（デフォルト）")
    logger.log("  2: 2フレームおき")
    logger.log("  3: 3フレームおき")
    logger.log("  5: 5フレームおき")
    frame_skip = get_user_input("フレーム間引き", default="1", value_type=int)
    logger.log(f"フレーム間引き: {frame_skip}")
    
    if frame_skip < 1:
        frame_skip = 1

    # === Step 6: 出力先設定 ===
    logger.log("\n【Step 6】出力先設定")
    logger.log("-" * 40)

    default_output_dir = os.path.join(os.path.dirname(video_path), "calibration_output")
    output_dir = get_user_input("出力ディレクトリ", default=default_output_dir, value_type=str)
    logger.log(f"出力ディレクトリ: {output_dir}")

    # 引用符を除去
    output_dir = output_dir.strip('"').strip("'")

    # === 設定確認 ===
    logger.log("\n" + "=" * 60)
    logger.log("設定内容の確認")
    logger.log("=" * 60)
    logger.log("\n【入力】")
    logger.log(f"  動画ファイル: {video_path}")
    logger.log("\n【レンズタイプ】")
    logger.log(f"  {lens_label}")
    logger.log("\n【チェスボード】")
    logger.log(f"  サイズ: {rows}行 x {cols}列（交点数）")
    logger.log(f"  マスサイズ: {square_size} m")
    logger.log("\n【キャリブレーション設定】")
    logger.log(f"  目標フレーム数: {target_frame_count}")
    logger.log(f"  ブレ判定の厳しさ: {blur_threshold}")
    logger.log("  画面分割: 分割なし")


    logger.log("\n【その他】")
    logger.log(f"  フレーム間引き: {frame_skip}{'（全フレーム処理）' if frame_skip == 1 else 'フレームおき'}")
    logger.log("\n【出力先】")
    logger.log(f"  {output_dir}")

    if not get_yes_no("\nこの設定でキャリブレーションを実行しますか？", default=True):
        logger.log("この設定でキャリブレーションを実行しますか？: No")
        logger.log("\nキャンセルしました。")
        return None
    logger.log("この設定でキャリブレーションを実行しますか？: Yes")

    # === キャリブレーション実行 ===
    logger.log("\n" + "=" * 60)
    logger.log(f"キャリブレーション実行中...（{lens_label}・分割なしモード）")
    logger.log("=" * 60 + "\n")

    os.makedirs(output_dir, exist_ok=True)

    if is_fisheye:
        calibrator = FisheyeMethodJ(
            checkerboard_rows=rows,
            checkerboard_cols=cols,
            square_size=square_size,
            target_frame_count=target_frame_count,
            blur_threshold=blur_threshold,
            enable_k_center=enable_k_center,
            frame_skip=frame_skip,
            logger=logger  # ロガーを渡す
        )
    else:
        calibrator = MethodJ_GeometricDiversity(
            checkerboard_rows=rows,
            checkerboard_cols=cols,
            square_size=square_size,
            target_frame_count=target_frame_count,
            blur_threshold=blur_threshold,
            enable_k_center=enable_k_center,
            frame_skip=frame_skip,
            logger=logger  # ロガーを渡す
        )

    start_time = time.time()

    try:
        # progress_callback は print を使うので、これを書き換えるか、あるいは
        # MethodJ の _log を使うように変更するか。
        # progress_callback は utils.py にあり print するだけ。
        # MethodJ は _log を使っているので、progress_callback は None にして
        # _log だけで進捗を表示するようにする手もあるが、従来の % 表示も捨てがたい
        # 今回は progress_callback はそのままで（標準出力のみ）、
        # 重要なログは logger 経由で process_log に入るため良しとする。
        # ただし "Unify" の要望なら progress もログに残すべきか？
        # 通常プログレスバー的なものはログに残さない方が良い。
        
        result = calibrator.run_calibration(
            video_path,
            progress_callback=progress_callback
        )
    except Exception as e:
        logger.log(f"\nエラー: キャリブレーション失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

    elapsed_time = time.time() - start_time

    # === 結果表示 ===
    logger.log("\n" + "=" * 60)
    logger.log(f"キャリブレーション完了！（{lens_label}・分割なし）")
    logger.log("=" * 60)
    logger.log("\n結果:")
    logger.log(f"  誤差（小さいほど高精度）: {result['rms']:.4f}")
    logger.log(f"  使用フレーム数: {len(result['per_view_errors'])}")
    logger.log(f"  処理時間: {elapsed_time:.2f}秒")
    logger.log(f"  95%タイル誤差（大半のフレームの誤差上限）: {np.percentile(result['per_view_errors'], 95):.4f}")

    logger.log("\nカメラ内部パラメータ:")
    logger.log(str(result['camera_matrix']))

    logger.log("\nレンズ歪み補正値:")
    logger.log(str(result['dist_coeffs']))
    if is_fisheye:
        logger.log("  ※ 魚眼モデル（4パラメータ）")

    # === ファイル保存 ===
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    prefix = "fisheye_calibration" if is_fisheye else "calibration"

    # npzファイル保存 (テキストも保存されるようになった)
    npz_filename = f"{prefix}_{timestamp}.npz"
    npz_path = os.path.join(output_dir, npz_filename)
    calibrator.save_calibration(result, npz_path)
    logger.log("\n保存しました:")
    logger.log(f"  キャリブレーションファイル: {npz_path}")
    logger.log(f"  テキスト形式レポート: {npz_path.replace('.npz', '_result.txt')}")

    # ログ保存（DualLoggerの全履歴を保存）
    log_filename = f"{prefix}_log_{timestamp}.txt"
    log_path = os.path.join(output_dir, log_filename)
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(logger.get_logs()))
    print(f"  処理ログ: {log_path}") # ここは print で OK

    logger.log("\n" + "=" * 60)
    logger.log("処理が完了しました。")
    logger.log("=" * 60)

    return result


