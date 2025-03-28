概要
YoloBokehEffectは、YOLOv5を使用した物体検出とHaar Cascadeによる顔検出を組み合わせたC++プロジェクトです。
このプログラムは、動画内の人物を検出し、その顔に**ボケ効果（ぼかし）**を適用することで、背景を強調します。
特徴
•	YOLOv5を用いた人物検出
•	Haar Cascadeを用いた顔検出
•	顔部分にぼかし効果を適用
必要条件
•	OpenCVライブラリ
•	YOLOv5モデル（yolov5s.onnx）
•	Haar Cascade顔検出XMLファイル（haarcascade_frontalface_default.xml）
使用方法
1.	リポジトリをクローンします。
2.	動画ファイル、YOLOv5モデル、Haar Cascade XMLをDataフォルダに配置します。
3.	コードをコンパイルして実行します。
4.	顔がぼかされた出力を表示します。
ライセンス
このプロジェクトはMITライセンスのもとで提供されています。
