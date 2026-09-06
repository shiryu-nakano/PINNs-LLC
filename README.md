# spatially_varying_diffusion
A Physics-Informed Neural Network based inverse solver for spatially varying diffusion


# 注意事項
```
nakano@s29:~$ cd server/
nakano@s29:~/server$ ls
dsb_llc  hf_cache  pinns_llc
nakano@s29:~/server$ mkdir PINNs-LLC
nakano@s29:~/server$ ls
PINNs-LLC  dsb_llc  hf_cache  pinns_llc
nakano@s29:~/server$ cd PINNs-LLC/
nakano@s29:~/server/PINNs-LLC$ vim README.md
nakano@s29:~/server/PINNs-LLC$ cat README.md
以下のディレクトリに，保存すべきすべてのデータをおく
計算結果，計算に必要なデータのダウンロード，重み，などすべて．
メインのリポジトリには，実行ようのスクリプト以外は基本置かない．
ここはデータサーバで容量は大きいが，リポジトリを置いている場所はGPUサーバで，容量に余裕がないため．
データのダウンロードやチェックポイントの保存はここに置かなければ途中で実行が止まる可能性もあるため注意．
nakano@s29:~/server/PINNs-LLC$ pwd
/home/nakano/server/PINNs-LLC
nakano@s29:~/server/PINNs-LLC$
```

上記はGPU1，でs29と呼ぶ．
以下はGPU2，s23と呼ぶ．ここからも同様にデータサーバにアクセスできる
```

nakano@s23:~$ cd server/
nakano@s23:~/server$ ls
PINNs-LLC  dsb_llc  hf_cache  pinns_llc
nakano@s23:~/server$

```

いかに，スクリプトが保存されているディレクトリをまとめておく
s29,s23ともに同じパス管理が可能なようにしてある．
```

nakano@s29:~/src/github.com/shiryu-nakano$ cd
DeepLearning_tutorial/ dotfiles/              llc_reproduction/
PINNs-LLC/             features-across-time/
nakano@s29:~/src/github.com/shiryu-nakano$ cd PINNs-LLC/
nakano@s29:~/src/github.com/shiryu-nakano/PINNs-LLC$ pwd
/home/nakano/src/github.com/shiryu-nakano/PINNs-LLC
nakano@s29:~/src/github.com/shiryu-nakano/PINNs-LLC$


nakano@s23:~/src/github.com/shiryu-nakano/PINNs-LLC$ pwd
/home/nakano/src/github.com/shiryu-nakano/PINNs-LLC
nakano@s23:~/src/github.com/shiryu-nakano/PINNs-LLC$


```
