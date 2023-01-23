# セッションベース推薦システムの実装
|モデル名|論文へのURL|状態表現|
|--|--|--|
|GRU4Rec|https://arxiv.org/abs/1511.06939|GRU|
|BCQ|https://arxiv.org/abs/1812.02900|GRU|
|SQN (SAC)|https://arxiv.org/abs/2006.05779|GRU|
|DQN|https://arxiv.org/abs/1509.06461|GRU|
|SDAC|https://ojs.aaai.org/index.php/AAAI/article/view/16579|GRU|
|QRDQN|https://arxiv.org/abs/1710.10044|GRU|

## 結果
- RecSysChallenge 2015 Yoochooseデータによる実験
    - クリック履歴と購入履歴のデータを持つ
- 前処理
    - セッションの長さが1以下のセッションを削除
    - 購入データ内に存在するセッションのみを抽出
    - 20000セッションをサンプリング
    - セッション数:20000, アイテム数: 15874, クリック数: 209020, 購入数: 62345
- 評価方法
    - クリックと購入についてそれぞれHit, NDCGによる評価を行う

クリック
|モデル名|Hit@10|NDCG@10|
|--|--|--|
|GRU4Rec|0.406|0.344|
|BCQ|0.254|0.249|
|SQN|0.420|0.359|
|SAC|0.227|0.181|
|DQN|0.00196|0.00171|
|QRDQN(BC)|0.229|0.190|

購入
|モデル名|Hit@10|NDCG@10|
|--|--|--|
|GRU4Rec|0.536|0.443|
|BCQ|0.347|0.328|
|SQN|0.545|0.449|
|SAC|0.382|0.306|
|DQN|0.00254|0.00175|
|QRDQN(BC)|0.390|0.339|
