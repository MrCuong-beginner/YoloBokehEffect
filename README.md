YoloBokehEffect

概要

YoloBokehEffectは、YOLOv5を使用した物体検出とHaar Cascadeによる顔検出を組み合わせたC++プロジェクトです。
このプログラムは、動画内の人物を検出し、その顔に**ボケ効果 (ぼかし)**を適用することで、背景を強調します。

特徴

YOLOv5を用いた人物検出

Haar Cascadeを用いた顔検出

顔部分にぼかし効果を適用

必要条件

OpenCVライブラリ

YOLOv5モデル (Data/yolov5s.onnx)

Haar Cascade顔検出XMLファイル (Data/haarcascade_frontalface_default.xml)

使用方法

リポジトリをクローンします。

動画ファイル (例: Data/PeopleVideo.mp4)、YOLOv5モデル、Haar Cascade XMLをDataフォルダに配置します。

コードをコンパイルして実行します。

動画の中の人物を検出し、顔の部分にボケ効果を適用します。

結果をリアルタイムで表示します。

'q' キーを押すと、プログラムが終了します。

ライセンス

このプロジェクトはMITライセンスのもとで提供されています。

