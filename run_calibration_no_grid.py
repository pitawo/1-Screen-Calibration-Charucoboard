"""
カメラキャリブレーション実行スクリプト（分割なし専用）
画面分割なしで、スケール・角度の多様性に基づくフレーム選択を行う
通常レンズ・魚眼レンズ（GoPro等）対応
"""
import os
import sys
import time
from datetime import datetime as dt

def main():
    """メインエントリーポイント（分割なし専用）"""
    import argparse

    parser = argparse.ArgumentParser(
        description='カメラキャリブレーション（分割なし） - 通常レンズ・魚眼レンズ対応',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 通常レンズ
  python run_calibration_no_grid.py video.mp4
  python run_calibration_no_grid.py video.mp4 --rows 5 --cols 8 --square_size 0.025

  # 魚眼レンズモード
  python run_calibration_no_grid.py video.mp4 --fisheye

※ 画面分割あり（2x2, 3x3）で実行する場合は run_calibration.py を使用してください
        """
    )

    # 引数がない場合は対話モードを実行
    if len(sys.argv) == 1:
        from system import interactive_mode_no_grid
        interactive_mode_no_grid()
        return

    parser.add_argument('video', nargs='?', help='キャリブレーション用動画のパス')
    parser.add_argument('--output_dir', '-o',
                        default='./output',
                        help='出力ディレクトリ（デフォルト: ./output）')
    parser.add_argument('--rows', type=int, default=5,
                        help='チェスボード行数（交点数）（デフォルト: 5）')
    parser.add_argument('--cols', type=int, default=8,
                        help='チェスボード列数（交点数）（デフォルト: 8）')
    parser.add_argument('--square_size', type=float, default=0.025,
                        help='マスのサイズ [m]（デフォルト: 0.025）')
    parser.add_argument('--target_frames', type=int, default=200,
                        help='目標フレーム数（デフォルト: 200）')
    parser.add_argument('--blur_threshold', type=float, default=120.0,
                        help='ブレ判定の厳しさ。高いほど厳しい（デフォルト: 120.0）')
    parser.add_argument('--no_k_center', action='store_true',
                        help='フレーム自動補完を無効化')
    parser.add_argument('--frame_skip', type=int, default=1,
                        help='フレーム間引き（Nフレームおきに処理。メモリ節約用。デフォルト: 1）')
    parser.add_argument('--fisheye', action='store_true',
                        help='魚眼レンズモード（GoPro等の広角レンズ用）')
    parser.add_argument('--preview', action='store_true',
                        help='キャリブレーション前にプレビューを表示')

    args = parser.parse_args()

    from system import (
        MethodJ_GeometricDiversity,
        FisheyeMethodJ,
        get_video_info,
        progress_callback,
        show_chessboard_preview,
        get_yes_no,
        DualLogger,
    )

    # ロガー初期化
    logger = DualLogger()

    if not args.video:
        parser.error("動画ファイルを指定してください（例: python run_calibration_no_grid.py video.mp4）")

    # 動画存在確認
    if not os.path.exists(args.video):
        logger.log(f"エラー: 動画ファイルが見つかりません: {args.video}")
        sys.exit(1)

    # プレビュー
    if args.preview:
        preview_output_dir = os.path.dirname(args.video) or "."
        show_chessboard_preview(args.video, args.rows, args.cols, output_dir=preview_output_dir)
        if not get_yes_no("\nキャリブレーションを続行しますか？", default=True):
            logger.log("キャンセルしました。")
            sys.exit(0)

    # 出力ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)

    # 動画情報表示
    video_info = get_video_info(args.video)
    lens_label = "魚眼レンズ" if args.fisheye else "通常レンズ"

    logger.log("=" * 60)
    logger.log(f"カメラキャリブレーション - {lens_label}（分割なし）")
    logger.log("=" * 60)
    logger.log("\n動画情報:")
    logger.log(f"  ファイル: {args.video}")
    logger.log(f"  解像度: {video_info['width']}x{video_info['height']}")
    logger.log(f"  FPS: {video_info['fps']:.2f}")
    logger.log(f"  フレーム数: {video_info['frame_count']}")

    logger.log("\nチェスボードパラメータ:")
    logger.log(f"  行数（交点）: {args.rows}")
    logger.log(f"  列数（交点）: {args.cols}")
    logger.log(f"  マスサイズ: {args.square_size} m")

    logger.log("\nキャリブレーションパラメータ:")
    logger.log(f"  レンズタイプ: {lens_label}")
    logger.log(f"  目標フレーム数: {args.target_frames}")
    logger.log(f"  ブレ判定の厳しさ: {args.blur_threshold}")
    logger.log("  画面分割: 分割なし")

    logger.log("\n" + "-" * 60)
    logger.log("キャリブレーション開始...")
    logger.log("-" * 60 + "\n")

    # キャリブレーター作成（grid_size=1 固定、min_frames_per_edge=0）
    if args.fisheye:
        calibrator = FisheyeMethodJ(
            checkerboard_rows=args.rows,
            checkerboard_cols=args.cols,
            square_size=args.square_size,
            target_frame_count=args.target_frames,
            blur_threshold=args.blur_threshold,
            enable_k_center=not args.no_k_center,
            frame_skip=args.frame_skip,
            logger=logger
        )
    else:
        calibrator = MethodJ_GeometricDiversity(
            checkerboard_rows=args.rows,
            checkerboard_cols=args.cols,
            square_size=args.square_size,
            target_frame_count=args.target_frames,
            blur_threshold=args.blur_threshold,
            enable_k_center=not args.no_k_center,
            frame_skip=args.frame_skip,
            logger=logger
        )

    start_time = time.time()

    try:
        # 非対話モードでは progress_callback は標準出力のみに出すのが通例だが
        # logには残さない方が綺麗かもしれない。しかし要望は "terminal output ... same as log file output"
        # なので、厳密には残すべきだが、進捗バーをログに残すと汚くなる。
        # ここでは progress_callback はそのままで（標準出力のみ）、
        # 重要なログは logger 経由で記録される。
        result = calibrator.run_calibration(
            args.video,
            progress_callback=progress_callback
        )
    except Exception as e:
        logger.log(f"\nエラー: キャリブレーション失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed_time = time.time() - start_time

    import numpy as np

    # 結果表示
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
    if args.fisheye:
        logger.log("  ※ 魚眼モデル（4パラメータ）")

    # 保存
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    prefix = "fisheye_calibration" if args.fisheye else "calibration"
    npz_filename = f"{prefix}_{timestamp}.npz"
    npz_path = os.path.join(args.output_dir, npz_filename)

    calibrator.save_calibration(result, npz_path)
    logger.log(f"\n結果を保存しました: {npz_path}")
    logger.log(f"テキスト形式レポート: {npz_path.replace('.npz', '_result.txt')}")

    # ログ保存（Unified Logging）
    log_filename = f"{prefix}_log_{timestamp}.txt"
    log_path = os.path.join(args.output_dir, log_filename)
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(logger.get_logs()))
    print(f"ログを保存しました: {log_path}")


if __name__ == '__main__':
    main()
