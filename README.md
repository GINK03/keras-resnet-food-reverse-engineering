# Resnetで料理の材料をあてていく:food2stuff

## モチベーション
- ResNetを使ってみたい
- 強い転移学習方法を教わったのでやってみたい
- img2cal(画像からカロリーを当てるタスク)などに転用できるかもと思った

## food2stuffって？
- stuffって材料って意味らしいです
- 画像からなんの構成品目でできており、なんの食材でできているかベクトル表現で出力します
- 2048次元の出力に対応しておりメジャーどころの食材はだいたい抑えています
- 将来的には自分が食べたものをinstagramなどで写真を取るだけで栄養素の管理とかできたらいいよね

## 学習用のデータ・セット集め
- Cookpadさんのサイトから95万件のレシピと投稿写真をあつめさせていただきました  
 レシピは個人の自由な書き方が許さており、林檎という表現一つとってもリンゴ、りんご、林檎と3つぐらい激しく揺らぐのですが、ここで小細工を入れるということを昔はしていましたが、一応の王道はデータ量で押し切ることです  
  それでどうにもこうにも行かなくなったら、いろいろな仕組みの導入を検討すべきというのが最近の私のやり方です  
  スクレイパーは[自作のもの](https://github.com/GINK03/kotlin-phantomjs-selenium-jsoup-parser)を用いました。
- 頻出するレシピの単語を上位2048個以外削る  
 出力次元数をどこまでも高次元にするのはGPUのメモリに乗らなくなるなど、現実的でないので、2048個に限定します  

## 学習するネットワークの選定
- ディープであるのは決定なのですが、Inception, ResNet, VGG16などいろいろあり、ResNetを今回は使おうと思いました。  
 TensorFlowなどではinception系の資料が豊富で学習のやりかたも幾つか確立されており、マニュアルに従うだけでかんたんに学習できます  
 VGGなども軽くていいのですが、画像識別で強い方を見ているとResNetが良さそうだと思いました。  
　
 ## ResNet転移学習ハックを適応する
 - この前のCookpadの[画像識別コンテスト](https://info.cookpad.com/pr/news/press_2017_0110)で好成績を収めていた方が使っていた手法が、BatchNormalizationレイヤーを再トレーニングするという転移学習の方法でした  
  この方法を採用して、BatchNormalizationレイヤのフリーズはしません
  Python3 + keras2で行いました   　
 ```python
  model = Model(inputs=resnet_model.input, outputs=result)
  for layer in model.layers[:139]: # default 179
    if 'BatchNormalization' in str(layer):
      ...
    else:
      layer.trainable = False
  model.compile(loss='binary_crossentropy', optimizer='adam')
  return model
 ```
 - StochasticDepthを適応する  
  StochasticDepthは今回、twitterでResNetの転移学習でいい方法ないかなと聞いたところ、だんごさんからアドバイスいただいた方法です  
  具体的にはResNetは何層もの直列のアンサンブルにしたもので、miniBatchごとに確率的にLayerをスキップします  
  なかなか安定しなく、安定した実装ができたら公開したいと思います  
 <p align="center">
   <img src="https://cloud.githubusercontent.com/assets/4949982/25310735/6f8e8094-2827-11e7-96db-539011d6f717.png">
 </p>
 <div align="center"> 図1. Stochasit Depth </div>
 
 ## 集めた画像をResNetの入力値にサイズにリサイズ
 めんどくさいですが、正しくやる必要があります。
 PILを利用して、縮小して、四角形の何もない画像に貼り付けました。
 ```python
 def cookpad():
  target_size = (224,224)
  dir_path = "../kotlin-phantomjs-selenium-jsoup-parser/dataset/*.jpg"
  max_size = len(glob.glob(dir_path))
  for i, name in enumerate(glob.glob(dir_path)):
    if i%10 == 0:
      print(i, max_size, name)
    save_name = name.split("/")[-1]
    if Path("cookpad/imgs/{save_name}.minify".format(save_name=save_name)).is_file():
      continue
    try:
      img = Image.open(name)
    except OSError as e:
      continue
    w, h = img.size
    if w > h :
      blank = Image.new('RGB', (w, w))
    if w <= h :
      blank = Image.new('RGB', (h, h))
    try:
      blank.paste(img, (0, 0) )
    except OSError as e:
      continue
    blank = blank.resize( target_size )
    blank.save("cookpad/imgs/{save_name}.mini.jpeg".format(save_name=save_name), "jpeg" )
 ```
 
 ## 学習
 いよいよ学習です。自宅のゲーム用に買ったマシンをLinuxとWindowsのデュアルブートにしてあるものがあるのですが、Windowsはここ半年ぐらい起動してません(何故買ったし)  
 GTX 1070でおこない、100epoch回すのに、20時間ほど必要でした。  
 ```
 $ python3 deep_food.py --train
 ```
 
 ##  過学習の取り扱い
  明確に正解を定義する必要があるので、その分が実装大変になるのですが、epochごとのmodelをダンプしてモデルに未知の画像を投入して、一番良いepochを定性的に決めるということをしました。  
  90epoch前後のモデルが良いと判断しました。 
 
 ## 学習済みモデルのダウンロード
 dropboxにおいてあります  
 ```
 $ wget https://www.dropbox.com/s/tgeosjt4i5dg79b/model00099.model
 $ mv model00099.model models/
 ```
 
 ## 予想タスク
 　ネットを徘徊して訓練に用いたデータであるCookpad以外のサイトから幾つか集めました。  
   そして評価した結果です。  
 ```sh
 $ python3 deep_food.py --pred
 ```
   特徴が少なく、突飛でない料理に関しては良好な結果です。  
<p align="center">
  <img width="700px" src="https://cloud.githubusercontent.com/assets/4949982/25310624/e60b1218-2823-11e7-8a5b-430628cd05b0.png">
</p>
<div align="center"> 図1. プリン </div>

<p align="center">
  <img width="700px" src="https://cloud.githubusercontent.com/assets/4949982/25310628/070214f8-2824-11e7-96e0-a88f75004197.png">
</p>
<div align="center"> 図2. パスタ </div>

<p align="center">
  <img width="700px" src="https://cloud.githubusercontent.com/assets/4949982/25310633/241c9dc4-2824-11e7-8405-0ea6bb2fbd43.png">
</p>
<div align="center"> 図3. クッキー </div>

<p align="center">
  <img width="700px" src="https://cloud.githubusercontent.com/assets/4949982/25310639/35bc12c6-2824-11e7-8f3d-d0f6a75d434e.png">
</p>
<div align="center"> 図4. チャーハン </div>

<p align="center">
  <img width="700px" src="https://cloud.githubusercontent.com/assets/4949982/25310643/474764fa-2824-11e7-9906-a18a09d6866c.png">
</p>
<div align="center"> 図5. カレー </div>

<p align="center">
  <img width="700px" src="https://cloud.githubusercontent.com/assets/4949982/25310645/588e0d2c-2824-11e7-8ab9-6d1fc5b2ec0b.png">
</p>
<div align="center"> 図6. サラダ </div>

<p align="center">
  <img width="700px" src="https://cloud.githubusercontent.com/assets/4949982/25310648/62c91bc4-2824-11e7-9623-3db3302b1f58.png">
</p>
<div align="center"> 図7. GOGOカレー（失敗例：揚げ物と勘違いしている） </div>

## 今回の実装
非商用・研究目的の場合自由にご利用ください  
[GitHub](https://github.com/GINK03/keras-resnet-food-reverse-engineering)

 ## 参考文献
 [1] [Deep Networks with Stochastic Depth](http://www.eccv2016.org/files/posters/S-3A-08.pdf)  
 [2] [Stochastic Depth元論文](https://arxiv.org/pdf/1603.09382.pdf)
