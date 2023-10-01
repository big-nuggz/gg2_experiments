# 使用方法
## 環境設定
### 必要なモジュールのインストール
CUDA11.7用のPyTorch 1.13.1を次のコマンドでインストールします．

`pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117`

次に，requrement.txtを使用して他の必要モジュールをインストールします．

`pip install -r requirements.txt`

### GaitGraph2のファイル
本リポジトリに含まれていないファイルを[GaitGraph2のリポジトリ](https://github.com/tteepe/GaitGraph2)からコピーしてくる必要があります．

必要なのはGaitGraph/modelsとGaitGraph/transformsです．その2つのフォルダをルートにコピーしてください．

以上で実行に必要な環境が構築できたはずです．

## データセットについて
学習用データは本実装では以下のような構造を想定しています．

フォルダ名はdataset_root以外は指定した名づけ方でないと動きません．番号は01から2桁の通し番号です．

```
dataset_root/
  config.txt
  cam01/
    subject01/
      session01.csv
```

config.txtは，データセット構造を示す数値が改行区切りで書かれています．例えば，カメラ2台，被験者10名，セッション5つなら，次の通りの内容になります．

```
2
10
5
```

各セッションのcsvファイルは，動画から抽出した骨格データを記録したものになります．1列目が8桁のフレーム番号，残りがx座標，y座標，可視性を各ランドマークごとにまとめて並べたものになっています．

私の作成したデータセットと，学習済みモデルの重みは[ここ](https://drive.google.com/file/d/1DYimtE9sthd6sCafacy7Y19MIrhP5188/view?usp=sharing)からダウンロードできます．学習済み重みは事前学習をせずに私が作成したデータセットのみで学習を行ったものなので，精度はいまいちです．

## 実行方法
現時点ではCPUのみでの実行となっております．GPUでの動作については後ほど実装の予定です．

2つのファイル，gaitgraph2_train.pyとgaitgraph2_test.pyを使用して学習とテストが行えます．私の作成したデータセットをダウンロードしてあれば，それぞれそのまま実行できるはすです．

gaitgraph2_train.pyはソース内で指定されたデータセットを用いてGaitGraph2モデルの学習を行います．現時点で学習中のvalidationは行っていないので，学習データでlossが最小になった重みをbest.pthの名で保存します．

gaitgraph2_test.pyは学習したモデルを使用する方法のサンプルになります．テスト内では，subject01のsession01の特徴量を比較用データとし，全subjectのsession02の特徴量と比較した結果（特徴ベクトル同士の距離）を出力しています．

デフォルトの比較対象はsubject01ですが，ファイル内の

```
# prepare test sequences
files = [
    f'{dataset_path}/cam01/subject01/session01.csv', # gallery
    f'{dataset_path}/cam01/subject01/session02.csv', 
    f'{dataset_path}/cam01/subject02/session02.csv', 
    f'{dataset_path}/cam01/subject03/session02.csv', 
    f'{dataset_path}/cam01/subject04/session02.csv', 
    f'{dataset_path}/cam01/subject05/session02.csv', 
    f'{dataset_path}/cam01/subject06/session02.csv', 
    f'{dataset_path}/cam01/subject07/session02.csv', 
    f'{dataset_path}/cam01/subject08/session02.csv', 
    f'{dataset_path}/cam01/subject09/session02.csv', 
    f'{dataset_path}/cam01/subject10/session02.csv', 
    f'{dataset_path}/cam01/subject11/session02.csv', 
    f'{dataset_path}/cam01/subject12/session02.csv', 
]
```

この部分の内，galleryとコメントでマークされた行を別のsubjectのファイルに変更することで，比較対象のIDを変更することができます．

## モデルの入力形式について
骨格構造に関しては，デフォルトでMediaPipeのものを使用する設定になっています．datasets/graph.pyに新たな構造を追記すれば，他の構造も使用できます．その際，モデル初期化時のオプション辞書の中の，graph_nameを対応するグラフ構造の名前に変更してください．

モデルの入力データは，PyTorch GeometricのDataオブジェクトです．xに60フレーム分の骨格データをTensor行列として格納したものになります．入力データの詳しい作り方はgaitgraph2_test.pyを参照してください．

データは，モデルに入力する前に，トランスフォームを適用する必要があります．これは，gaitgraph2_modelのGaitGraph2Transformsクラスから得られます．

## 出力ベクトルと距離の計算
詳しくはgaitgraph2_test.pyの距離計算部分を参照してください．モデルからはTensorのベクトルが出力されるので，それをギャラリーベクトルとtorch.cdist()を使用して比較すれば距離が算出できます．