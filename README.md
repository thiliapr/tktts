# Tk TTS
## 简介
这是一个自由的语音生成软件，主要使用[librosa](https://librosa.org/)来解析音频，[pytorch](https://github.com/pytorch/pytorch/)来机器学习，旨在通过深度学习技术生成音乐作品。

## License
![GNU AGPL Version 3 Logo](https://www.gnu.org/graphics/agplv3-with-text-162x68.png)

TkTTS 是自由软件，遵循`Affero GNU 通用公共许可证第 3 版或任何后续版本`。你可以自由地使用、修改和分发该软件，但不提供任何明示或暗示的担保。有关详细信息，请参见 [Affero GNU 通用公共许可证](https://www.gnu.org/licenses/agpl-3.0.html)。

## 使用示例
这里演示的是大致流程，实际可能需要调整，不过一般照着这个来就行了

### 安装依赖
#### CPU 用户
```bash
pip install -r requirements.txt
```

#### CUDA 用户（Nvidia 显卡）
```bash
nvidia-smi  # 查看 CUDA 版本
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
```

> [!TIP]
> 根据`nvidia-smi`的输出`CUDA Version`把`cu128`换成你自己的 CUDA 版本，比如输出`CUDA Version: 12.1`就把`cu128`替换为`cu126`  
> 具体来说，PyTorch 的CUDA是向下兼容的，所以选择时只需要选择比自己的 CUDA 版本小一点的版本就行了。  
> 比如 PyTorch 提供了三个版本: `12.6, 12.8, 12.9`，然后你的 CUDA 版本是`12.7`，那么就选择`12.8`（因为官方提供的`12.6` < 你的`12.7` < 官方提供的`12.8`）

### 准备数据集
准备一个数据集，比如我用游戏提取脚本提取游戏语音作为数据集，这里演示使用 The LJ Speech Dataset 数据集

#### 预处理一般数据集
```bash
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2  # https://keithito.com/LJ-Speech-Dataset/
```
然后写一个脚本，将数据集转化为这种结构：对于每一个音频，都应该有一个对应的元数据文件
```plaintext
dataset
+---select_oblige
|   \---kuk
|           fem_kuk_00001.ogg
|           fem_kuk_00001.ogg.json
|
\---senren_banka
        akh108_001.ogg
        akh108_001.ogg.json
```
而元数据内容应类似这样
```json
{
    "text":"んー……文化部総括としての報告は、大してない。\n活動報告は同じく資料にまとめてあるから、それで",
    "positive_prompt":["source:セレクトオブリージュ","sex:female","age:teenager","voice_actor:相模恋","character:夜刀くくる"],
    "negative_prompt":["voice_actor:秋野花","character:鞍馬小春","character:獅童龍司","source:千恋＊万花","sex:male","voice_actor:風音","character:常陸茉子","character:コーチ","voice_actor:桜川未央","age:adult","voice_actor:由嘉鈍","voice_actor:七種結花","voice_actor:奏雨","voice_actor:沢澤砂羽","character:駒川みづは","voice_actor:飴川紫乃","character:星継銀音","character:ファイブ","voice_actor:椨もんじゃ","voice_actor:真宮ゆず","character:Ｋ子","character:一色奏命","character:中条比奈実","voice_actor:佐藤みかん","character:大屋汐莉","character:トウリ","character:猪谷心子","character:成宮帝雄","character:朝武秋穂","voice_actor:小鳥居夕花","voice_actor:天知遥","character:バアさん","age:old","voice_actor:北大路ゆき","character:モンステラ","voice_actor:北見六花","voice_actor:木住葵","character:朝武安晴","character:ムラサメ","voice_actor:ナオト†サンクチュアリ","character:北条花","character:葦華真智","voice_actor:碓氷珊瑚","character:レナ·リヒテナウアー","voice_actor:山崎高","character:Ｕ子","voice_actor:ちとせ杏","source:アイコトバ -Silver Snow Sister-","character:西山冴希","voice_actor:白砂菓夏海","character:一ノ瀬七","age:children","character:北条空","character:蓼科イヴ","voice_actor:東シヅ","character:朝武芳乃","character:鞍馬玄十郎","voice_actor:御苑生メイ","character:鞍馬廉太郎","voice_actor:遥そら"]
}
```


#### 从 Artemis 游戏中提取数据集
1. 使用[GARbro](https://github.com/crskycode/GARbro)从游戏目录的`xxx.pfs`文件提取出`sound/vo`和`script`文件夹，分别保存到`/path/to/game/sound/vo`和`/path/to/game/script`
2. 运行`python extract_artemis.py /path/to/game/script /path/to/game/sound/vo /path/to/artemis_pre_dataset`，它会输出一个角色ID对应的角色名次数
3. 仿照`examples/select_oblige_c2t.json`和`examples/aikotoba_sss_c2t.json`，根据第二步的输出，写你要提取的游戏的角色ID-角色名映射表，保存到`/path/to/artemis_c2t.json`
4. 运行`python convert_artemis_to_dataset.py /path/to/artemis_pre_dataset /path/to/artemis_c2t.json /path/to/artemis_dataset -p source:<游戏名> -p <其他你想在所有对话加上的标签> -n <你想在所有对话加上的负面标签>`
5. 你的数据集应该已经在`/path/to/artemis_dataset`了

#### 从 Kirikiri Z 游戏中提取数据集
1. 使用[GARbro](https://github.com/crskycode/GARbro)分别从游戏目录的`data.xp3`文件和`voice.xp3`提取出`scn`和根目录文件夹，分别保存到`/path/to/game/script`和`/path/to/game/voice`
2. 运行`python extract_kirikiriz.py /path/to/game/script /path/to/game/voice /path/to/kirikiriz_pre_dataset`，它会输出所有对话出现的角色名
3. 仿照`examples/senren_banka_c2t.json`，根据第二步的输出，写你要提取的游戏的角色ID-角色名映射表，保存到`/path/to/kirikiriz_c2t.json`
4. 运行`python convert_kirikiriz_to_dataset.py /path/to/kirikiriz_pre_dataset /path/to/kirikiriz_c2t.json /path/to/kirikiriz_dataset -p source:<游戏名> -p <其他你想在所有对话加上的标签> -n <你想在所有对话加上的负面标签>`
5. 你的数据集应该已经在`/path/to/kirikiriz_dataset`了

### 可选: 对数据集进行提示词增强
我们都知道，有一些标签是不可能同时存在的，比如`sex:male`和`sex:female`就不可能同时出现在一个正面提示里，所以我们可以定义一堆标签互斥组，类似这样
```python
[
    {"sex:male", "sex:female"},
    {"age:children", "age:teenager", "age:adult", "age:old"},
    {"character:白上フブキ", "character:夏色まつり"}  # 举这两个例子是因为我喜欢看
]
```
对于一个音频的元数据，我们遍历每一个标签互斥组（类型：集合），然后检测元数据的正面标签是否包含且仅包含了这个标签互斥组中的一个标签，比如
```python
positive_prompt = {"age:children"}
negative_prompt: set[str]
groups = [{"age:children", "age:teenager", "age:adult", "age:old"}, ...]
for group in groups:
    if len(intersection := positive_prompt & group) == 1:  # 为什么不是只要检测到包含就全部加入负面提示呢？因为可能有些用户想要同时指定两个冲突的标签；鬼知道他们是怎么想的
        negative_prompt.update(group - intersection)  # 将该组其余所有的互斥标签加入负面提示
```
我写了一个脚本来节省自己造轮子的麻烦，你可以通过运行
```bash
python augment_prompt_with_exclusion.py /path/to/dataset /path/to/dataset /path/to/mutually_exclusive_groups.json
```
来实现提示词增强
> ![TIP]
> 参数的两个`/path/to/dataset`是为了省提示增强后需要手动将改写后的`post_dataset`复制回`/path/to/dataset`的麻烦
> 如果你想，你也可以指定一个`/path/to/dataset_pre`和`/path/to/dataset_post`，但这麻烦且毫无意义
> 此处`/path/to/mutually_exclusive_groups.json`内容请参考`examples/mutually_exclusive_groups.json`

### 训练分词器
```bash
python train_tokenizer.py /path/to/ckpt -t /path/to/train_dataset -v /path/to/val_dataset
```

### 初始化检查点
```bash
python list_tags_from_datasets.py /path/to/dataset -o /path/to/tags.txt
python init_checkpoint.py /path/to/ckpt -t /path/to/tags.txt
```

### 训练模型
```bash
python train_tktts.py <num_epochs> /path/to/ckpt -t /path/to/train_dataset -v /path/to/val_dataset
```
将`<num_epochs>`替换为实际的你想训练的轮数  
> [!TIP]
> 你可以在训练后运行`python show_scales.py /path/to/ckpt`来看看每层的缩放因子，按数据流向排序

### 生成
```bash
python list_tags_from_ckpt.py /path/to/ckpt  # 列出所有标签
python generate.py /path/to/ckpt output.wav ホロライブ所属のバーチャルyoutuber、夏色まつりだよっ！！ -p character:夏色まつり -p source:youtube -p sex:female -p age:teenager -n sex:male
```

### 注意事项
- 请在命令行输入`python3 file.py --help`获得帮助

## 文档
文档是不可能写的，这辈子都不可能写的。经验表明，写了文档只会变成“代码一天一天改，文档一年不会动”的局面，反而误导人。

所以我真心推荐：有什么事直接看代码（代码的注释和函数的文档还是会更新的），或者复制代码问ai去吧（记得带上下文）。

## 贡献与开发
欢迎提出问题、改进或贡献代码。如果有任何问题或建议，您可以在 GitHub 上提 Issues，或者直接通过电子邮件联系开发者。

## 联系信息
如有任何问题或建议，请联系项目维护者 thiliapr。
- Email: thiliapr@tutanota.com

# 无关软件本身的广告
## Join the Blue Ribbon Online Free Speech Campaign!
![Blue Ribbon Campaign Logo](https://www.eff.org/files/brstrip.gif)

支持[Blue Ribbon Online 言论自由运动](https://www.eff.org/pages/blue-ribbon-campaign)！  
你可以通过向其[捐款](https://supporters.eff.org/donate)以表示支持。

## 支持自由软件运动
为什么要自由软件: [GNU 宣言](https://www.gnu.org/gnu/manifesto.html)

你可以通过以下方式支持自由软件运动:
- 向非自由程序或在线敌服务说不，哪怕只有一次，也会帮助自由软件。不和其他人使用它们会帮助更大。进一步，如果你告诉人们这是在捍卫自己的自由，那么帮助就更显著了。
- [帮助 GNU 工程和自由软件运动](https://www.gnu.org/help/help.html)
- [向 FSF 捐款](https://www.fsf.org/about/ways-to-donate/)